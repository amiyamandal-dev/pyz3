/// Example module to benchmark GIL optimization
const std = @import("std");
const py = @import("pyz3");

pub fn nested_allocations() !u64 {
    // This function triggers multiple nested allocations
    // With GIL caching, only the first allocation acquires GIL
    const allocator = py.allocator;

    const buffer1 = try allocator.alloc(u8, 1024);
    defer allocator.free(buffer1);

    const buffer2 = try allocator.alloc(u8, 2048);
    defer allocator.free(buffer2);

    const buffer3 = try allocator.alloc(u8, 512);
    defer allocator.free(buffer3);

    return 3; // Number of allocations completed
}

pub fn deep_nesting() !u64 {
    // Create a deep call stack with allocations
    return try recursive_alloc(10);
}

fn recursive_alloc(depth: u64) !u64 {
    if (depth == 0) return 0;

    const allocator = py.allocator;
    const buffer = try allocator.alloc(u8, 128);
    defer allocator.free(buffer);

    const result = try recursive_alloc(depth - 1);
    return result + depth;
}

pub fn benchmark_allocations(args: struct { iterations: u64 }) !u64 {
    // Benchmark memory allocation performance
    const allocator = py.allocator;
    const iterations = args.iterations;

    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        const buffer = try allocator.alloc(u8, 256);
        defer allocator.free(buffer);
    }

    return iterations; // Return number of iterations completed
}

pub fn container_allocations() !py.PyList(@This()) {
    // Create containers that allocate memory internally
    const list = try py.PyList(@This()).new(0);

    var i: u64 = 0;
    while (i < 100) : (i += 1) {
        const num = try py.PyLong.create(i);
        try list.append(num.obj);
    }

    return list;
}

comptime {
    py.rootmodule(@This());
}

test "nested allocations" {
    py.initialize();
    defer py.finalize();

    const result = try nested_allocations();
    try std.testing.expectEqual(@as(u64, 3), result);
}

test "deep nesting" {
    py.initialize();
    defer py.finalize();

    const result = try deep_nesting();
    try std.testing.expectEqual(@as(u64, 55), result); // Sum of 1..10
}

test "benchmark allocations" {
    py.initialize();
    defer py.finalize();

    const result = try benchmark_allocations(.{ .iterations = 1000 });
    try std.testing.expectEqual(@as(u64, 1000), result);
}
