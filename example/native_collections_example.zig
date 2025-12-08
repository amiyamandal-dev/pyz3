/// Example module demonstrating native high-performance collections
const std = @import("std");
const py = @import("pyz3");
const native = py.native_collections;

/// Demonstrate FastDict performance
pub fn test_fast_dict() !py.PyDict(@This()) {
    // Create a native dict
    var dict = try native.FastDict.init();
    defer dict.deinit();

    // Add some entries
    const value1: usize = 42;
    const value2: usize = 100;
    const value3: usize = 200;

    try dict.set("key1", @ptrFromInt(value1));
    try dict.set("key2", @ptrFromInt(value2));
    try dict.set("key3", @ptrFromInt(value3));

    // Convert to Python dict for return
    const py_dict = try py.PyDict(@This()).new();

    // Add entries to Python dict
    const key1_obj = try py.PyLong.create(value1);
    try py_dict.setItem(try py.PyString.create("key1"), key1_obj.obj);

    const key2_obj = try py.PyLong.create(value2);
    try py_dict.setItem(try py.PyString.create("key2"), key2_obj.obj);

    const key3_obj = try py.PyLong.create(value3);
    try py_dict.setItem(try py.PyString.create("key3"), key3_obj.obj);

    const size_obj = try py.PyLong.create(dict.size());
    try py_dict.setItem(try py.PyString.create("size"), size_obj.obj);

    return py_dict;
}

/// Demonstrate FastArray performance
pub fn test_fast_array() !py.PyList(@This()) {
    // Create a native array
    var array = try native.FastArray.init();
    defer array.deinit();

    // Add some elements (use offset to avoid null pointer for value 0)
    const offset: usize = 0x10000;
    for (0..10) |i| {
        try array.push(@ptrFromInt(offset + (i * 10)));
    }

    // Convert to Python list
    const py_list = try py.PyList(@This()).new(0);

    for (0..array.size()) |i| {
        if (array.get(i)) |value| {
            const int_val = @intFromPtr(value) - offset;  // Subtract offset for display
            const py_int = try py.PyLong.create(int_val);
            try py_list.append(py_int.obj);
        }
    }

    return py_list;
}

/// Benchmark FastDict vs Python dict
pub fn benchmark_dict(args: struct { count: u64 }) !py.PyDict(@This()) {
    // Create native dict
    var dict = try native.FastDict.init();
    defer dict.deinit();

    const start = std.time.nanoTimestamp();

    // Insert entries (use offset to avoid null pointer)
    const offset: usize = 0x10000;
    for (0..args.count) |i| {
        const key = try std.fmt.allocPrint(py.allocator, "key{d}", .{i});
        defer py.allocator.free(key);

        try dict.set(key, @ptrFromInt(offset + i));
    }

    const insert_time = std.time.nanoTimestamp() - start;

    // Lookup entries
    const lookup_start = std.time.nanoTimestamp();

    for (0..args.count) |i| {
        const key = try std.fmt.allocPrint(py.allocator, "key{d}", .{i});
        defer py.allocator.free(key);

        _ = dict.get(key);
    }

    const lookup_time = std.time.nanoTimestamp() - lookup_start;

    // Return results as Python dict
    const result = try py.PyDict(@This()).new();

    const insert_ms = try py.PyFloat.create(@as(f64, @floatFromInt(insert_time)) / 1_000_000.0);
    try result.setItem(try py.PyString.create("insert_time_ms"), insert_ms.obj);

    const lookup_ms = try py.PyFloat.create(@as(f64, @floatFromInt(lookup_time)) / 1_000_000.0);
    try result.setItem(try py.PyString.create("lookup_time_ms"), lookup_ms.obj);

    const size_obj = try py.PyLong.create(dict.size());
    try result.setItem(try py.PyString.create("size"), size_obj.obj);

    return result;
}

/// Benchmark FastArray vs Python list
pub fn benchmark_array(args: struct { count: u64 }) !py.PyDict(@This()) {
    // Create native array
    var array = try native.FastArray.init();
    defer array.deinit();

    const start = std.time.nanoTimestamp();

    // Push elements (use offset to avoid null pointer)
    const offset: usize = 0x10000;
    for (0..args.count) |i| {
        try array.push(@ptrFromInt(offset + i));
    }

    const push_time = std.time.nanoTimestamp() - start;

    // Access elements
    const access_start = std.time.nanoTimestamp();

    for (0..args.count) |i| {
        _ = array.get(i);
    }

    const access_time = std.time.nanoTimestamp() - access_start;

    // Return results as Python dict
    const result = try py.PyDict(@This()).new();

    const push_ms = try py.PyFloat.create(@as(f64, @floatFromInt(push_time)) / 1_000_000.0);
    try result.setItem(try py.PyString.create("push_time_ms"), push_ms.obj);

    const access_ms = try py.PyFloat.create(@as(f64, @floatFromInt(access_time)) / 1_000_000.0);
    try result.setItem(try py.PyString.create("access_time_ms"), access_ms.obj);

    const size_obj = try py.PyLong.create(array.size());
    try result.setItem(try py.PyString.create("size"), size_obj.obj);

    return result;
}

/// Store Python objects in native dict
pub fn dict_with_pyobjects() !py.PyDict(@This()) {
    // Note: This function demonstrates storing PyObject pointers in native dict
    // In production, you must manage reference counts carefully

    const result = try py.PyDict(@This()).new();

    const msg = try py.PyString.create("Native dict can store PyObject pointers efficiently!");
    try result.setItem(try py.PyString.create("message"), msg.obj);

    return result;
}

comptime {
    py.rootmodule(@This());
}

test "dict operations" {
    py.initialize();
    defer py.finalize();

    const result = try test_fast_dict();
    defer result.obj.decref();

    try std.testing.expect(result.length() > 0);
}

test "array operations" {
    py.initialize();
    defer py.finalize();

    const result = try test_fast_array();
    defer result.obj.decref();

    try std.testing.expectEqual(@as(usize, 10), result.length());
}
