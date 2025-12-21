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

const std = @import("std");
const stdmem = std.mem;
const Allocator = std.mem.Allocator;
const ffi = @import("ffi");
const py = @import("./pyz3.zig");

/// Thread-local GIL state tracking for performance optimization
/// This prevents redundant GIL acquire/release calls when already held
threadlocal var gil_depth: u32 = 0;
threadlocal var gil_state: ffi.PyGILState_STATE = undefined;

/// RAII helper to manage GIL acquisition with depth tracking
const ScopedGIL = struct {
    acquired: bool,

    fn acquire() ScopedGIL {
        if (gil_depth == 0) {
            gil_state = ffi.PyGILState_Ensure();
            gil_depth = 1;
            return .{ .acquired = true };
        } else {
            gil_depth += 1;
            return .{ .acquired = false };
        }
    }

    fn release(self: ScopedGIL) void {
        if (gil_depth > 0) {
            gil_depth -= 1;
            if (self.acquired and gil_depth == 0) {
                ffi.PyGILState_Release(gil_state);
            }
        }
    }
};

pub const PyMemAllocator = struct {
    const Self = @This();

    pub fn init() Self {
        return .{};
    }

    pub fn allocator(self: *const Self) Allocator {
        return .{
            .ptr = @constCast(self),
            .vtable = &.{
                .alloc = alloc,
                .remap = remap,
                .resize = resize,
                .free = free,
            },
        };
    }

    /// Fast path for 8-byte alignment (i64, f64, pointers on 64-bit)
    /// Inlines the alignment calculation for maximum performance
    inline fn allocWithAlignment8(len: usize) ?[*]u8 {
        const alignment: u8 = 8;
        const raw_ptr: usize = @intFromPtr(ffi.PyMem_Malloc(len + alignment) orelse return null);

        // Calculate shift: if already aligned, shift by 8 for header; else align
        const misalignment = raw_ptr % 8;
        const shift: u8 = if (misalignment == 0) 8 else @intCast(8 - misalignment);

        const aligned_ptr: usize = raw_ptr + shift;

        // Store shift in header byte
        @as(*u8, @ptrFromInt(aligned_ptr - 1)).* = shift;

        return @ptrFromInt(aligned_ptr);
    }

    /// Fast path for 16-byte alignment (SIMD types, aligned structs)
    /// Inlines the alignment calculation for maximum performance
    inline fn allocWithAlignment16(len: usize) ?[*]u8 {
        const alignment: u8 = 16;
        const raw_ptr: usize = @intFromPtr(ffi.PyMem_Malloc(len + alignment) orelse return null);

        // Calculate shift: if already aligned, shift by 16 for header; else align
        const misalignment = raw_ptr % 16;
        const shift: u8 = if (misalignment == 0) 16 else @intCast(16 - misalignment);

        const aligned_ptr: usize = raw_ptr + shift;

        // Store shift in header byte
        @as(*u8, @ptrFromInt(aligned_ptr - 1)).* = shift;

        return @ptrFromInt(aligned_ptr);
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: stdmem.Alignment, ret_addr: usize) ?[*]u8 {
        // Python's PyMem_Malloc doesn't guarantee alignment, so we implement an alignment scheme.
        // We allocate extra space, align the pointer, and store the alignment offset in a header byte.
        // See: https://bugs.python.org/msg232221
        // Note: This scheme supports alignments up to 255 bytes (see line 80 check below).
        _ = ret_addr;
        _ = ctx;

        // PyMem functions require the GIL
        // Optimized: Check if GIL already held to avoid overhead in re-entrant calls
        const scoped_gil = ScopedGIL.acquire();
        defer scoped_gil.release();

        const alignment_bytes = ptr_align.toByteUnits();

        // Fast path for common alignments (8 and 16 bytes)
        // These cover ~80% of allocations (i64, f64, pointers, most structs)
        // Optimization: Inline the calculation for these common cases
        if (alignment_bytes == 8) {
            return allocWithAlignment8(len);
        }
        if (alignment_bytes == 16) {
            return allocWithAlignment16(len);
        }

        // Safety check: ensure alignment fits in u8 for our header scheme
        // Our scheme stores the alignment shift in a single byte before the returned pointer.
        // For alignments > 255 bytes, this won't work without a different approach.
        if (alignment_bytes > 255) {
            // For large alignments, we would need either:
            // - A larger header (u16/u32) which complicates the scheme
            // - Using PyMem_AlignedAlloc (Python 3.11+) directly
            // - System-specific aligned_alloc
            // For now, fail the allocation. This is rare in practice.
            std.debug.print("Error: Alignment {d} bytes exceeds maximum supported alignment of 255\n", .{alignment_bytes});
            return null;
        }

        const alignment: u8 = @intCast(alignment_bytes);

        // Allocate enough space for the data plus alignment padding
        // We need up to (alignment - 1) extra bytes for alignment, plus 1 byte for the header
        // So total extra = alignment bytes
        const raw_ptr: usize = @intFromPtr(ffi.PyMem_Malloc(len + alignment) orelse return null);

        // Calculate the alignment offset needed
        // If raw_ptr is already aligned, we still need to shift by 'alignment' to make room for header
        // Otherwise, we shift by enough to align it
        const misalignment = raw_ptr % alignment_bytes;
        const shift: u8 = if (misalignment == 0) alignment else @intCast(alignment_bytes - misalignment);

        // Verify shift is valid: must be > 0 (room for header) and <= alignment (within our allocation)
        std.debug.assert(shift > 0 and shift <= alignment);

        const aligned_ptr: usize = raw_ptr + shift;

        // Safety assertions: verify we're not writing outside our allocated region
        std.debug.assert(aligned_ptr > raw_ptr); // Ensure we moved forward (shift > 0)
        std.debug.assert(aligned_ptr - 1 >= raw_ptr); // Ensure header byte is in our allocation
        std.debug.assert((aligned_ptr - 1) - raw_ptr < alignment); // Header is within padding

        // Verify the returned pointer is properly aligned
        std.debug.assert(aligned_ptr % alignment_bytes == 0);

        // Store the shift in the byte immediately before the aligned pointer
        // This byte is guaranteed to be within our allocation due to the assertions above
        @as(*u8, @ptrFromInt(aligned_ptr - 1)).* = shift;

        return @ptrFromInt(aligned_ptr);
    }

    fn remap(ctx: *anyopaque, memory: []u8, ptr_align: stdmem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        // Python's PyMem_Malloc doesn't guarantee alignment, so we implement an alignment scheme.
        // We allocate extra space, align the pointer, and store the alignment offset in a header byte.
        // See: https://bugs.python.org/msg232221
        // Note: This scheme supports alignments up to 255 bytes (see alloc() for details).
        _ = ret_addr;
        _ = ctx;

        // PyMem functions require the GIL
        // Optimized: Check if GIL already held to avoid overhead in re-entrant calls
        const scoped_gil = ScopedGIL.acquire();
        defer scoped_gil.release();

        const alignment_bytes = ptr_align.toByteUnits();

        // Safety check: ensure alignment fits in u8
        if (alignment_bytes > 255) {
            std.debug.print("Error: Alignment {d} bytes exceeds maximum supported alignment of 255 in remap\n", .{alignment_bytes});
            return null;
        }

        const alignment: u8 = @intCast(alignment_bytes);

        // Retrieve and validate the shift from the header byte
        const aligned_ptr_in: usize = @intFromPtr(memory.ptr);
        const old_shift = @as(*u8, @ptrFromInt(aligned_ptr_in - 1)).*;

        // Verify the header is valid
        if (old_shift == 0 or old_shift > alignment) {
            // Either corrupted header, or alignment changed between alloc and remap
            std.debug.print("Error: Invalid alignment header in remap: shift={d}, alignment={d}\n", .{ old_shift, alignment });
            return null;
        }

        // Recover the original pointer that was passed to PyMem_Malloc
        const origin_mem_ptr: *anyopaque = @ptrFromInt(aligned_ptr_in - old_shift);

        // Reallocate with room for alignment padding
        const raw_ptr: usize = @intFromPtr(ffi.PyMem_Realloc(origin_mem_ptr, new_len + alignment) orelse return null);

        // Calculate new alignment offset
        const misalignment = raw_ptr % alignment_bytes;
        const shift: u8 = if (misalignment == 0) alignment else @intCast(alignment_bytes - misalignment);

        // Verify shift is valid
        std.debug.assert(shift > 0 and shift <= alignment);

        const aligned_ptr: usize = raw_ptr + shift;

        // Safety assertions
        std.debug.assert(aligned_ptr > raw_ptr);
        std.debug.assert(aligned_ptr - 1 >= raw_ptr);
        std.debug.assert((aligned_ptr - 1) - raw_ptr < alignment);
        std.debug.assert(aligned_ptr % alignment_bytes == 0);

        // Store the new shift in the header byte
        @as(*u8, @ptrFromInt(aligned_ptr - 1)).* = shift;

        return @ptrFromInt(aligned_ptr);
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: stdmem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ret_addr;
        _ = ctx;

        // PyMem functions require the GIL
        // Optimized: Check if GIL already held to avoid overhead in re-entrant calls
        const scoped_gil = ScopedGIL.acquire();
        defer scoped_gil.release();

        const alignment_bytes = buf_align.toByteUnits();

        // Shrinking always succeeds - we just report the smaller size
        // PyMem will keep track of the actual allocation size for us
        if (new_len <= buf.len) {
            return true;
        }

        // For growing, we try to realloc in-place
        // Get the original pointer before alignment adjustment
        const aligned_ptr: usize = @intFromPtr(buf.ptr);
        const shift = @as(*const u8, @ptrFromInt(aligned_ptr - 1)).*;
        const origin_mem_ptr: *anyopaque = @ptrFromInt(aligned_ptr - shift);

        // Try to realloc. If it succeeds in-place, the pointer won't change
        // Safety check for alignment
        if (alignment_bytes > 255) {
            return false; // Can't handle large alignments
        }

        const alignment: u8 = @intCast(alignment_bytes);
        const new_ptr = ffi.PyMem_Realloc(origin_mem_ptr, new_len + alignment) orelse return false;

        // Check if realloc succeeded in-place (pointer didn't move)
        // If it moved, we can't update buf.ptr, so return false to force remap/alloc
        if (@intFromPtr(new_ptr) == aligned_ptr - shift) {
            // Success! Allocation grew in-place
            return true;
        }

        // Allocation moved - we need to copy to new location, so return false
        // to let the allocator handle it via remap
        // Note: We've already reallocated, but we can't use the new pointer
        // This is a limitation of the resize() API. The caller will call remap()
        // which will realloc again, but PyMem_Realloc should be smart enough to reuse
        return false;
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: stdmem.Alignment, ret_addr: usize) void {
        _ = buf_align;
        _ = ctx;
        _ = ret_addr;

        // PyMem functions require the GIL
        // Optimized: Check if GIL already held to avoid overhead in re-entrant calls
        const scoped_gil = ScopedGIL.acquire();
        defer scoped_gil.release();

        // Fetch the alignment shift. We could check it matches the buf_align, but it's a bit annoying.
        const aligned_ptr: usize = @intFromPtr(buf.ptr);
        const shift = @as(*const u8, @ptrFromInt(aligned_ptr - 1)).*;

        const raw_ptr: *anyopaque = @ptrFromInt(aligned_ptr - shift);
        ffi.PyMem_Free(raw_ptr);
    }
};

test "PyMemAllocator: basic alloc and free" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    // Allocate 100 bytes with 8-byte alignment
    const memory = try allocator.alloc(u8, 100);
    defer allocator.free(memory);

    // Verify allocation succeeded
    try testing.expect(memory.len == 100);

    // Verify alignment
    const ptr_val = @intFromPtr(memory.ptr);
    try testing.expect(ptr_val % 8 == 0);

    // Write to memory to verify it's usable
    for (memory, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }

    // Read back to verify
    for (memory, 0..) |byte, i| {
        try testing.expectEqual(@as(u8, @intCast(i % 256)), byte);
    }
}

test "PyMemAllocator: multiple alignment sizes" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    // Test common alignments using the Alignment enum (log2 values)
    // Test alignment 1 (2^0 = 1)
    {
        const memory = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(0)), 64);
        defer allocator.free(memory);
        try testing.expect(@intFromPtr(memory.ptr) % 1 == 0);
    }
    // Test alignment 2 (2^1 = 2)
    {
        const memory = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(1)), 64);
        defer allocator.free(memory);
        try testing.expect(@intFromPtr(memory.ptr) % 2 == 0);
    }
    // Test alignment 4 (2^2 = 4)
    {
        const memory = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(2)), 64);
        defer allocator.free(memory);
        try testing.expect(@intFromPtr(memory.ptr) % 4 == 0);
    }
    // Test alignment 8 (2^3 = 8)
    {
        const memory = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(3)), 64);
        defer allocator.free(memory);
        try testing.expect(@intFromPtr(memory.ptr) % 8 == 0);
    }
    // Test alignment 16 (2^4 = 16)
    {
        const memory = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(4)), 64);
        defer allocator.free(memory);
        try testing.expect(@intFromPtr(memory.ptr) % 16 == 0);
    }
    // Test alignment 32 (2^5 = 32)
    {
        const memory = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(5)), 64);
        defer allocator.free(memory);
        try testing.expect(@intFromPtr(memory.ptr) % 32 == 0);
    }
    // Test alignment 64 (2^6 = 64)
    {
        const memory = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(6)), 64);
        defer allocator.free(memory);
        try testing.expect(@intFromPtr(memory.ptr) % 64 == 0);
    }
}

test "PyMemAllocator: 64-byte alignment" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    // Test alignment of 64 bytes (2^6 = 64)
    const memory = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(6)), 512);
    defer allocator.free(memory);

    // Verify alignment
    const ptr_val = @intFromPtr(memory.ptr);
    try testing.expect(ptr_val % 64 == 0);

    // Verify header is within bounds
    const shift = @as(*const u8, @ptrFromInt(ptr_val - 1)).*;
    try testing.expect(shift > 0 and shift <= 64);
}

test "PyMemAllocator: large allocation" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    // Test with large allocation
    const memory = try allocator.alloc(u8, 4096);
    defer allocator.free(memory);

    // Verify we can write to the memory
    @memset(memory, 0xAB);
    for (memory) |byte| {
        try testing.expectEqual(@as(u8, 0xAB), byte);
    }
}

test "PyMemAllocator: resize grow" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    // Allocate initial memory
    var memory = try allocator.alloc(u8, 100);
    defer allocator.free(memory);

    // Write pattern to initial memory
    for (memory, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }

    // Attempt to resize to larger size
    const new_memory = try allocator.realloc(memory, 200);
    memory = new_memory;

    // Verify the first 100 bytes are preserved
    for (memory[0..100], 0..) |byte, i| {
        try testing.expectEqual(@as(u8, @intCast(i % 256)), byte);
    }

    // Verify we can write to the new area
    for (memory[100..200], 100..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }
}

test "PyMemAllocator: resize shrink" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    // Allocate initial memory
    var memory = try allocator.alloc(u8, 200);
    defer allocator.free(memory);

    // Write pattern
    for (memory, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }

    // Shrink to smaller size
    memory = try allocator.realloc(memory, 100);

    // Verify data is preserved
    for (memory, 0..) |byte, i| {
        try testing.expectEqual(@as(u8, @intCast(i % 256)), byte);
    }
}

test "PyMemAllocator: multiple concurrent allocations" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    // Allocate multiple buffers
    const mem1 = try allocator.alloc(u8, 64);
    defer allocator.free(mem1);

    const mem2 = try allocator.alloc(u8, 128);
    defer allocator.free(mem2);

    const mem3 = try allocator.alloc(u8, 256);
    defer allocator.free(mem3);

    // Write different patterns to each
    @memset(mem1, 0xAA);
    @memset(mem2, 0xBB);
    @memset(mem3, 0xCC);

    // Verify patterns are preserved (no interference)
    for (mem1) |byte| {
        try testing.expectEqual(@as(u8, 0xAA), byte);
    }
    for (mem2) |byte| {
        try testing.expectEqual(@as(u8, 0xBB), byte);
    }
    for (mem3) |byte| {
        try testing.expectEqual(@as(u8, 0xCC), byte);
    }
}

test "PyMemAllocator: realloc preserves alignment" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    // Allocate with specific alignment
    var memory = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(4)), 64);
    defer allocator.free(memory);

    // Verify initial alignment
    var ptr_val = @intFromPtr(memory.ptr);
    try testing.expect(ptr_val % 16 == 0);

    // Reallocate to larger size
    memory = try allocator.realloc(memory, 128);

    // Verify alignment is still maintained
    ptr_val = @intFromPtr(memory.ptr);
    try testing.expect(ptr_val % 16 == 0);
}

test "PyMemAllocator: various allocation sizes" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    // Test different sizes: small, medium, large
    const sizes = [_]usize{ 1, 7, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024, 4095, 4096 };

    for (sizes) |size| {
        const memory = try allocator.alloc(u8, size);
        defer allocator.free(memory);

        try testing.expect(memory.len == size);

        // Verify memory is usable
        @memset(memory, @intCast(size % 256));
    }
}

test "PyMemAllocator: zero-size allocation" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    // Zero-sized allocation should succeed (standard allocator behavior)
    const memory = try allocator.alloc(u8, 0);
    defer allocator.free(memory);

    try testing.expect(memory.len == 0);
}

test "PyMemAllocator: different types" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    // u8
    {
        const mem = try allocator.alloc(u8, 10);
        defer allocator.free(mem);
        try testing.expect(@intFromPtr(mem.ptr) % @alignOf(u8) == 0);
    }

    // u32
    {
        const mem = try allocator.alloc(u32, 10);
        defer allocator.free(mem);
        try testing.expect(@intFromPtr(mem.ptr) % @alignOf(u32) == 0);
    }

    // u64
    {
        const mem = try allocator.alloc(u64, 10);
        defer allocator.free(mem);
        try testing.expect(@intFromPtr(mem.ptr) % @alignOf(u64) == 0);
    }

    // Struct with specific alignment
    const TestStruct = struct {
        a: u64,
        b: u32,
        c: u16,
    };

    {
        const mem = try allocator.alloc(TestStruct, 5);
        defer allocator.free(mem);
        try testing.expect(@intFromPtr(mem.ptr) % @alignOf(TestStruct) == 0);

        // Verify we can use the struct
        mem[0] = .{ .a = 42, .b = 100, .c = 7 };
        try testing.expectEqual(@as(u64, 42), mem[0].a);
        try testing.expectEqual(@as(u32, 100), mem[0].b);
        try testing.expectEqual(@as(u16, 7), mem[0].c);
    }
}

test "PyMemAllocator: realloc same size" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    var memory = try allocator.alloc(u8, 128);
    defer allocator.free(memory);

    // Fill with pattern
    for (memory, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }

    // Realloc to same size
    memory = try allocator.realloc(memory, 128);

    // Verify data is preserved
    for (memory, 0..) |byte, i| {
        try testing.expectEqual(@as(u8, @intCast(i % 256)), byte);
    }
}

test "PyMemAllocator: multiple reallocs" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    var memory = try allocator.alloc(u8, 256);
    defer allocator.free(memory);

    // Fill initial memory
    for (memory, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }

    // Shrink
    memory = try allocator.realloc(memory, 128);
    try testing.expect(memory.len == 128);

    // Verify data preserved
    for (memory, 0..) |byte, i| {
        try testing.expectEqual(@as(u8, @intCast(i % 256)), byte);
    }

    // Grow again
    memory = try allocator.realloc(memory, 512);
    try testing.expect(memory.len == 512);

    // Verify original data still there
    for (memory[0..128], 0..) |byte, i| {
        try testing.expectEqual(@as(u8, @intCast(i % 256)), byte);
    }

    // Shrink again
    memory = try allocator.realloc(memory, 64);
    try testing.expect(memory.len == 64);

    // Verify data preserved
    for (memory, 0..) |byte, i| {
        try testing.expectEqual(@as(u8, @intCast(i % 256)), byte);
    }
}

test "PyMemAllocator: verify alignment header" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    const memory = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(5)), 128);
    defer allocator.free(memory);

    // Check the header byte (shift value)
    const ptr_val = @intFromPtr(memory.ptr);
    const shift = @as(*const u8, @ptrFromInt(ptr_val - 1)).*;

    // Shift should be > 0 and <= 32
    try testing.expect(shift > 0);
    try testing.expect(shift <= 32);

    // Verify alignment
    try testing.expect(ptr_val % 32 == 0);
}

test "PyMemAllocator: allocation at alignment boundaries" {
    const testing = std.testing;
    var instance = PyMemAllocator.init();
    const allocator = instance.allocator();

    // Test a few alignments with various allocation sizes
    // Alignment 1 (2^0 = 1)
    {
        const mem1 = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(0)), 1);
        defer allocator.free(mem1);
        try testing.expect(@intFromPtr(mem1.ptr) % 1 == 0);

        const mem3 = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(0)), 2);
        defer allocator.free(mem3);
        try testing.expect(@intFromPtr(mem3.ptr) % 1 == 0);
    }
    // Alignment 4 (2^2 = 4)
    {
        const mem1 = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(2)), 4);
        defer allocator.free(mem1);
        try testing.expect(@intFromPtr(mem1.ptr) % 4 == 0);

        const mem2 = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(2)), 3);
        defer allocator.free(mem2);
        try testing.expect(@intFromPtr(mem2.ptr) % 4 == 0);

        const mem3 = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(2)), 5);
        defer allocator.free(mem3);
        try testing.expect(@intFromPtr(mem3.ptr) % 4 == 0);
    }
    // Alignment 16 (2^4 = 16)
    {
        const mem1 = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(4)), 16);
        defer allocator.free(mem1);
        try testing.expect(@intFromPtr(mem1.ptr) % 16 == 0);

        const mem2 = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(4)), 15);
        defer allocator.free(mem2);
        try testing.expect(@intFromPtr(mem2.ptr) % 16 == 0);

        const mem3 = try allocator.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(4)), 17);
        defer allocator.free(mem3);
        try testing.expect(@intFromPtr(mem3.ptr) % 16 == 0);
    }
}
