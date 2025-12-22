// Example demonstrating PySequence mixin usage
// Build with: zig build-exe sequence_example.zig

const std = @import("std");
const py = @import("pyz3");

const root = @This();

pub fn main() !void {
    py.initialize();
    defer py.finalize();

    std.debug.print("\n=== PySequence Mixin Examples ===\n\n", .{});

    try basicOperations();
    try searchOperations();
    try concatenationOperations();
    try iterationExamples();
    try functionalProgramming();
}

fn basicOperations() !void {
    std.debug.print("1. Basic Operations:\n", .{});

    var list = try py.PyList(root).new(5);
    defer list.obj.decref();

    // Set items
    try list.setItem(0, 10);
    try list.setItem(1, 20);
    try list.setItem(2, 30);
    try list.setItem(3, 40);
    try list.setItem(4, 50);

    // Get length
    const len = try list.len();
    std.debug.print("   Length: {}\n", .{len});

    // Get items (supports negative indices)
    const first = try list.getItem(i64, 0);
    const last = try list.getItem(i64, -1);
    std.debug.print("   First: {}, Last: {}\n", .{ first, last });

    // Check if empty
    const empty = try list.isEmpty();
    std.debug.print("   Is empty: {}\n", .{empty});

    // Get slice
    var slice = try list.getSlice(1, 4);
    defer slice.obj.decref();
    const slice_len = try slice.len();
    std.debug.print("   Slice [1:4] length: {}\n", .{slice_len});

    std.debug.print("\n", .{});
}

fn searchOperations() !void {
    std.debug.print("2. Search & Membership:\n", .{});

    var list = try py.PyList(root).new(5);
    defer list.obj.decref();

    try list.setItem(0, 10);
    try list.setItem(1, 20);
    try list.setItem(2, 10);
    try list.setItem(3, 30);
    try list.setItem(4, 10);

    // Check membership
    const has_20 = try list.contains(20);
    const has_99 = try list.contains(99);
    std.debug.print("   Contains 20: {}\n", .{has_20});
    std.debug.print("   Contains 99: {}\n", .{has_99});

    // Find index
    const idx = try list.index(20);
    std.debug.print("   Index of 20: {}\n", .{idx});

    // Count occurrences
    const count = try list.count(10);
    std.debug.print("   Count of 10: {}\n", .{count});

    std.debug.print("\n", .{});
}

fn concatenationOperations() !void {
    std.debug.print("3. Concatenation & Repetition:\n", .{});

    var list1 = try py.PyList(root).new(2);
    defer list1.obj.decref();
    try list1.setItem(0, 1);
    try list1.setItem(1, 2);

    var list2 = try py.PyList(root).new(2);
    defer list2.obj.decref();
    try list2.setItem(0, 3);
    try list2.setItem(1, 4);

    // Concatenate
    var concatenated = try list1.concat(list2);
    defer concatenated.obj.decref();
    const concat_len = try concatenated.len();
    std.debug.print("   Concatenated length: {}\n", .{concat_len});

    // Repeat
    var repeated = try list1.repeat(3);
    defer repeated.obj.decref();
    const repeat_len = try repeated.len();
    std.debug.print("   Repeated (x3) length: {}\n", .{repeat_len});

    // In-place concatenation
    var list3 = try py.PyList(root).new(2);
    defer list3.obj.decref();
    try list3.setItem(0, 10);
    try list3.setItem(1, 20);

    var list4 = try py.PyList(root).new(1);
    defer list4.obj.decref();
    try list4.setItem(0, 30);

    try list3.inplaceConcat(list4);
    const inplace_len = try list3.len();
    std.debug.print("   After in-place concat: {}\n", .{inplace_len});

    std.debug.print("\n", .{});
}

fn iterationExamples() !void {
    std.debug.print("4. Iteration:\n", .{});

    var list = try py.PyList(root).new(5);
    defer list.obj.decref();

    try list.setItem(0, 10);
    try list.setItem(1, 20);
    try list.setItem(2, 30);
    try list.setItem(3, 40);
    try list.setItem(4, 50);

    std.debug.print("   Items: ", .{});

    var iter = list.iterator();
    while (try iter.next()) |item| {
        defer item.decref();

        const val = try py.as(root, i64, item);
        std.debug.print("{} ", .{val});
    }

    std.debug.print("\n", .{});

    // Reset and iterate again
    iter.reset();
    var sum: i64 = 0;
    while (try iter.next()) |item| {
        defer item.decref();
        const val = try py.as(root, i64, item);
        sum += val;
    }
    std.debug.print("   Sum: {}\n", .{sum});

    std.debug.print("\n", .{});
}

fn functionalProgramming() !void {
    std.debug.print("5. Functional Programming:\n", .{});

    var list = try py.PyList(root).new(5);
    defer list.obj.decref();

    try list.setItem(0, 1);
    try list.setItem(1, 2);
    try list.setItem(2, 3);
    try list.setItem(3, 4);
    try list.setItem(4, 5);

    // Map: double each value
    const doubled = try list.map(struct {
        fn double(item: py.PyObject) py.PyObject {
            const val = py.as(root, i64, item) catch return item;
            return py.create(root, val * 2) catch item;
        }
    }.double);
    defer doubled.obj.decref();

    std.debug.print("   After map (double): ", .{});
    var iter = doubled.iterator();
    while (try iter.next()) |item| {
        defer item.decref();
        const val = try py.as(root, i64, item);
        std.debug.print("{} ", .{val});
    }
    std.debug.print("\n", .{});

    // Filter: keep only even values
    const evens = try list.filter(struct {
        fn isEven(item: py.PyObject) !bool {
            const val = try py.as(root, i64, item);
            return val % 2 == 0;
        }
    }.isEven);
    defer evens.obj.decref();

    std.debug.print("   After filter (evens): ", .{});
    var iter2 = evens.iterator();
    while (try iter2.next()) |item| {
        defer item.decref();
        const val = try py.as(root, i64, item);
        std.debug.print("{} ", .{val});
    }
    std.debug.print("\n", .{});

    std.debug.print("\n", .{});
}
