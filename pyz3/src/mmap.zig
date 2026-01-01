// Memory-mapped file utilities for pyz3.
//
// This module provides cross-platform memory-mapped file access,
// enabling zero-copy data sharing between Zig and Python.
//
// Licensed under the Apache License, Version 2.0

const std = @import("std");
const py = @import("pyz3.zig");
const builtin = @import("builtin");

/// Error types for mmap operations
pub const MmapError = error{
    OpenFailed,
    MmapFailed,
    InvalidSize,
    ReadOnly,
    FlushFailed,
    UnmapFailed,
};

/// Memory protection flags
pub const Protection = struct {
    read: bool = true,
    write: bool = false,
    execute: bool = false,

    pub fn toOsFlags(self: Protection) u32 {
        if (builtin.os.tag == .windows) {
            if (self.execute) {
                if (self.write) return 0x40; // PAGE_EXECUTE_READWRITE
                return 0x20; // PAGE_EXECUTE_READ
            }
            if (self.write) return 0x04; // PAGE_READWRITE
            return 0x02; // PAGE_READONLY
        } else {
            var flags: u32 = 0;
            if (self.read) flags |= std.posix.PROT.READ;
            if (self.write) flags |= std.posix.PROT.WRITE;
            if (self.execute) flags |= std.posix.PROT.EXEC;
            return flags;
        }
    }
};

/// Memory-mapped file handle
pub const MmapFile = struct {
    data: []align(std.mem.page_size) u8,
    len: usize,
    fd: std.posix.fd_t,
    is_readonly: bool,

    /// Open and memory-map an existing file
    pub fn open(path: []const u8, prot: Protection) !MmapFile {
        const flags: std.posix.O = if (prot.write)
            .{ .ACCMODE = .RDWR }
        else
            .{ .ACCMODE = .RDONLY };

        const fd = std.posix.open(path, flags, 0) catch {
            return MmapError.OpenFailed;
        };
        errdefer std.posix.close(fd);

        const stat = std.posix.fstat(fd) catch {
            return MmapError.OpenFailed;
        };

        const len: usize = @intCast(stat.size);
        if (len == 0) {
            return MmapError.InvalidSize;
        }

        const prot_flags = prot.toOsFlags();
        const data = std.posix.mmap(
            null,
            len,
            prot_flags,
            .{ .TYPE = .SHARED },
            fd,
            0,
        ) catch {
            return MmapError.MmapFailed;
        };

        return MmapFile{
            .data = data,
            .len = len,
            .fd = fd,
            .is_readonly = !prot.write,
        };
    }

    /// Create a new file and memory-map it
    pub fn create(path: []const u8, size: usize) !MmapFile {
        if (size == 0) {
            return MmapError.InvalidSize;
        }

        const fd = std.posix.open(
            path,
            .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true },
            0o644,
        ) catch {
            return MmapError.OpenFailed;
        };
        errdefer std.posix.close(fd);

        // Extend file to desired size
        std.posix.ftruncate(fd, @intCast(size)) catch {
            return MmapError.OpenFailed;
        };

        const data = std.posix.mmap(
            null,
            size,
            std.posix.PROT.READ | std.posix.PROT.WRITE,
            .{ .TYPE = .SHARED },
            fd,
            0,
        ) catch {
            return MmapError.MmapFailed;
        };

        return MmapFile{
            .data = data,
            .len = size,
            .fd = fd,
            .is_readonly = false,
        };
    }

    /// Get a typed slice view of the mapped memory
    pub fn asSlice(self: *const MmapFile, comptime T: type) []const T {
        const ptr: [*]const T = @ptrCast(@alignCast(self.data.ptr));
        return ptr[0 .. self.len / @sizeOf(T)];
    }

    /// Get a mutable typed slice view of the mapped memory
    pub fn asSliceMut(self: *MmapFile, comptime T: type) ![]T {
        if (self.is_readonly) {
            return MmapError.ReadOnly;
        }
        const ptr: [*]T = @ptrCast(@alignCast(self.data.ptr));
        return ptr[0 .. self.len / @sizeOf(T)];
    }

    /// Flush changes to disk
    pub fn flush(self: *MmapFile) !void {
        if (builtin.os.tag == .windows) {
            // Windows flush implementation would go here
            return;
        }

        const result = std.posix.msync(
            self.data,
            .{ .SYNC = true },
        );
        if (result != 0) {
            return MmapError.FlushFailed;
        }
    }

    /// Advise kernel about expected access pattern
    pub fn advise(self: *MmapFile, advice: Advice) void {
        if (builtin.os.tag == .linux or builtin.os.tag == .macos) {
            _ = std.posix.madvise(self.data, advice.toOsFlag());
        }
    }

    /// Close and unmap the file
    pub fn close(self: *MmapFile) void {
        std.posix.munmap(self.data);
        std.posix.close(self.fd);
        self.data = undefined;
        self.len = 0;
    }
};

/// Memory access pattern hints
pub const Advice = enum {
    normal,
    sequential,
    random,
    willneed,
    dontneed,

    fn toOsFlag(self: Advice) u32 {
        return switch (self) {
            .normal => std.posix.MADV.NORMAL,
            .sequential => std.posix.MADV.SEQUENTIAL,
            .random => std.posix.MADV.RANDOM,
            .willneed => std.posix.MADV.WILLNEED,
            .dontneed => std.posix.MADV.DONTNEED,
        };
    }
};

/// Create a shared memory region accessible from Python
pub fn createSharedBuffer(comptime T: type, count: usize) !struct {
    mmap: MmapFile,
    slice: []T,
} {
    const size = count * @sizeOf(T);
    const aligned_size = std.mem.alignForward(usize, size, std.mem.page_size);

    // Create anonymous mapping
    const data = std.posix.mmap(
        null,
        aligned_size,
        std.posix.PROT.READ | std.posix.PROT.WRITE,
        .{ .TYPE = .SHARED, .ANONYMOUS = true },
        -1,
        0,
    ) catch {
        return MmapError.MmapFailed;
    };

    const ptr: [*]T = @ptrCast(@alignCast(data.ptr));

    return .{
        .mmap = MmapFile{
            .data = data,
            .len = aligned_size,
            .fd = -1,
            .is_readonly = false,
        },
        .slice = ptr[0..count],
    };
}

/// Python-compatible wrapper for mmap operations
pub fn MmapWrapper(comptime root: type) type {
    return struct {
        const Self = @This();

        /// Load a binary file as a slice (zero-copy when possible)
        pub fn loadFile(args: struct {
            path: []const u8,
        }) !py.PyObject {
            _ = args;
            // Implementation would create a memoryview/buffer
            // that wraps the mmap'd data
            return py.PyNone(root).get();
        }

        /// Create a shared buffer accessible from Python
        pub fn createBuffer(args: struct {
            size: usize,
        }) !py.PyObject {
            _ = args;
            // Would return a memoryview wrapping shared memory
            return py.PyNone(root).get();
        }
    };
}

// Tests
test "mmap create and read" {
    const allocator = std.testing.allocator;
    _ = allocator;

    // Create temp file path
    const path = "/tmp/test_mmap.bin";

    // Create mmap file
    var mmap_file = try MmapFile.create(path, 4096);
    defer mmap_file.close();

    // Write data
    const data = try mmap_file.asSliceMut(u8);
    @memset(data, 0xAB);

    try mmap_file.flush();

    // Verify
    const read_data = mmap_file.asSlice(u8);
    try std.testing.expectEqual(@as(u8, 0xAB), read_data[0]);
    try std.testing.expectEqual(@as(u8, 0xAB), read_data[4095]);
}

test "mmap typed access" {
    const path = "/tmp/test_mmap_typed.bin";

    // Create with f64 data
    var mmap_file = try MmapFile.create(path, 8 * 100); // 100 f64s
    defer mmap_file.close();

    const floats = try mmap_file.asSliceMut(f64);
    for (floats, 0..) |*f, i| {
        f.* = @floatFromInt(i);
    }

    try mmap_file.flush();

    // Verify
    const read_floats = mmap_file.asSlice(f64);
    try std.testing.expectEqual(@as(f64, 0.0), read_floats[0]);
    try std.testing.expectEqual(@as(f64, 99.0), read_floats[99]);
}
