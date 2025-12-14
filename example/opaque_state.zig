// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// Example: Attaching Zig-Native State to Python Classes (Opaque to Python)
///
/// This demonstrates how to:
/// - Attach Zig allocators, pointers, and structs to Python objects
/// - Keep all internal state completely inaccessible from Python
/// - Safely clean up resources in __del__
/// - Prevent double-free and use-after-free

const std = @import("std");
const py = @import("pyz3");

/// Example 1: Simple opaque state with allocator and buffer
pub const BufferManager = py.class(struct {
    const Self = @This();
    pub const __doc__ = "Manages an internal buffer (opaque to Python)";

    // ✅ OPAQUE ZIG STATE - Completely invisible to Python
    allocator: std.mem.Allocator,
    buffer: []u8,
    size: usize,
    write_pos: usize,

    pub fn __init__(self: *Self, args: struct { size: usize }) !void {
        self.allocator = py.allocator;
        self.buffer = try self.allocator.alloc(u8, args.size);
        self.size = args.size;
        self.write_pos = 0;
    }

    pub fn write(self: *Self, args: struct { data: []const u8 }) !void {
        if (self.write_pos + args.data.len > self.size) {
            return py.ValueError(@This()).raise("Buffer overflow");
        }

        @memcpy(self.buffer[self.write_pos..][0..args.data.len], args.data);
        self.write_pos += args.data.len;
    }

    pub fn read_all(self: *Self) !py.PyBytes {
        return py.PyBytes.create(self.buffer[0..self.write_pos]);
    }

    pub fn clear(self: *Self) void {
        self.write_pos = 0;
    }

    pub fn __del__(self: *Self) void {
        self.allocator.free(self.buffer);
    }
});

/// Example 2: Complex Zig structures with multiple allocations
pub const DataProcessor = py.class(struct {
    const Self = @This();
    pub const __doc__ = "Processes data using complex internal Zig structures";

    // ✅ OPAQUE: Complex Zig-only data structures
    arena: std.heap.ArenaAllocator,
    work_buffer: []u8,
    result_buffer: []u8,
    process_count: usize,
    is_finalized: bool,  // Guard against double-free

    pub fn __init__(self: *Self, args: struct { buffer_size: usize }) !void {
        self.arena = std.heap.ArenaAllocator.init(py.allocator);
        const arena_alloc = self.arena.allocator();

        self.work_buffer = try arena_alloc.alloc(u8, args.buffer_size);
        self.result_buffer = try arena_alloc.alloc(u8, args.buffer_size);
        self.process_count = 0;
        self.is_finalized = false;
    }

    pub fn process(self: *Self, args: struct { data: []const u8 }) !py.PyBytes {
        if (self.is_finalized) {
            return py.RuntimeError(@This()).raise("Processor has been finalized");
        }

        // Use internal buffers
        const len = @min(args.data.len, self.work_buffer.len);
        @memcpy(self.work_buffer[0..len], args.data[0..len]);

        // Process: simple transformation (uppercase)
        for (self.work_buffer[0..len], 0..) |byte, i| {
            self.result_buffer[i] = if (byte >= 'a' and byte <= 'z')
                byte - 32
            else
                byte;
        }

        self.process_count += 1;
        return py.PyBytes.create(self.result_buffer[0..len]);
    }

    pub fn get_process_count(self: *Self) usize {
        return self.process_count;
    }

    pub fn finalize(self: *Self) void {
        if (!self.is_finalized) {
            self.arena.deinit();
            self.is_finalized = true;
        }
    }

    pub fn __del__(self: *Self) void {
        self.finalize();
    }
});

/// Example 3: Secure storage with encryption key (never exposed to Python)
pub const SecureStorage = py.class(struct {
    const Self = @This();
    pub const __doc__ = "Stores data securely with internal encryption";

    // ✅ OPAQUE: Security-critical data never exposed
    allocator: std.mem.Allocator,
    encryption_key: [32]u8,
    encrypted_data: []u8,
    data_length: usize,

    pub fn __init__(self: *Self) !void {
        self.allocator = py.allocator;
        self.encrypted_data = try self.allocator.alloc(u8, 0);
        self.data_length = 0;

        // Generate random encryption key - Python never sees this
        std.crypto.random.bytes(&self.encryption_key);
    }

    pub fn store(self: *Self, args: struct { data: []const u8 }) !void {
        // Reallocate if needed
        if (args.data.len > self.encrypted_data.len) {
            self.allocator.free(self.encrypted_data);
            self.encrypted_data = try self.allocator.alloc(u8, args.data.len);
        }

        // XOR "encryption" using internal key
        for (args.data, 0..) |byte, i| {
            self.encrypted_data[i] = byte ^ self.encryption_key[i % 32];
        }
        self.data_length = args.data.len;
    }

    pub fn retrieve(self: *Self) !py.PyBytes {
        // Decrypt and return
        const decrypted = try self.allocator.alloc(u8, self.data_length);
        defer self.allocator.free(decrypted);

        for (self.encrypted_data[0..self.data_length], 0..) |byte, i| {
            decrypted[i] = byte ^ self.encryption_key[i % 32];
        }

        return py.PyBytes.create(decrypted);
    }

    pub fn __del__(self: *Self) void {
        // Secure cleanup: zero out key before freeing
        @memset(&self.encryption_key, 0);
        self.allocator.free(self.encrypted_data);
    }
});

/// Example 4: File handle management
pub const FileManager = py.class(struct {
    const Self = @This();
    pub const __doc__ = "Manages file operations with internal file handle";

    // ✅ OPAQUE: File descriptor/handle never exposed
    file: ?std.fs.File,
    path_buffer: ?[]u8,
    allocator: std.mem.Allocator,
    bytes_written: usize,

    pub fn __init__(self: *Self) void {
        self.file = null;
        self.path_buffer = null;
        self.allocator = py.allocator;
        self.bytes_written = 0;
    }

    pub fn open(self: *Self, args: struct { path: []const u8 }) !void {
        // Close existing file if open
        if (self.file) |f| {
            f.close();
        }

        // Store path for error messages
        if (self.path_buffer) |buf| {
            self.allocator.free(buf);
        }
        self.path_buffer = try self.allocator.alloc(u8, args.path.len);
        @memcpy(self.path_buffer.?, args.path);

        // Open file
        self.file = try std.fs.cwd().createFile(args.path, .{});
        self.bytes_written = 0;
    }

    pub fn write(self: *Self, args: struct { data: []const u8 }) !void {
        if (self.file == null) {
            return py.ValueError(@This()).raise("No file opened");
        }

        try self.file.?.writeAll(args.data);
        self.bytes_written += args.data.len;
    }

    pub fn close(self: *Self) void {
        if (self.file) |f| {
            f.close();
            self.file = null;
        }
    }

    pub fn get_bytes_written(self: *Self) usize {
        return self.bytes_written;
    }

    pub fn __del__(self: *Self) void {
        self.close();
        if (self.path_buffer) |buf| {
            self.allocator.free(buf);
        }
    }
});

/// Example 5: Reference counting and shared state
pub const SharedResource = py.class(struct {
    const Self = @This();
    pub const __doc__ = "Resource with internal reference counting";

    // ✅ OPAQUE: Reference counting state
    allocator: std.mem.Allocator,
    resource: *Resource,

    pub fn __init__(self: *Self) !void {
        self.allocator = py.allocator;
        self.resource = try self.allocator.create(Resource);
        self.resource.* = .{ .ref_count = 1, .data = 0 };
    }

    pub fn increment_data(self: *Self) void {
        self.resource.data += 1;
    }

    pub fn get_data(self: *Self) i64 {
        return self.resource.data;
    }

    pub fn add_reference(self: *Self) void {
        self.resource.ref_count += 1;
    }

    pub fn __del__(self: *Self) void {
        self.resource.ref_count -= 1;
        if (self.resource.ref_count == 0) {
            self.allocator.destroy(self.resource);
        }
    }
});

const Resource = struct {
    ref_count: usize,
    data: i64,
};

comptime {
    py.rootmodule(@This());
}

// Tests
const testing = std.testing;

test "BufferManager opaque state" {
    py.initialize();
    defer py.finalize();

    var manager = try py.init(@This(), BufferManager, .{ .size = 100 });
    defer manager.__del__();

    try manager.write(.{ .data = "hello" });
    try manager.write(.{ .data = " world" });

    const result = try manager.read_all();
    defer result.obj.decref();

    const bytes = try result.asSlice();
    try testing.expectEqualStrings("hello world", bytes);
}

test "SecureStorage encryption" {
    py.initialize();
    defer py.finalize();

    var storage = try py.init(@This(), SecureStorage, .{});
    defer storage.__del__();

    const original = "secret message";
    try storage.store(.{ .data = original });

    const retrieved = try storage.retrieve();
    defer retrieved.obj.decref();

    const result = try retrieved.asSlice();
    try testing.expectEqualStrings(original, result);

    // Verify encrypted data is different from original
    try testing.expect(!std.mem.eql(u8, storage.encrypted_data[0..original.len], original));
}
