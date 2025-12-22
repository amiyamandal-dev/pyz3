// Example demonstrating NumPy integration in pyz3
// Shows how to use NumPy from Zig code

const std = @import("std");
const py = @import("pyz3");

const root = @This();

/// Create a NumPy array from Python list
pub fn create_array() !py.PyObject {
    const np = try py.numpy.getModule(@This());
    defer np.decref();

    // Create Python list
    const list = try py.PyList(root).new(0);
    errdefer list.obj.decref();

    try list.append(1.0);
    try list.append(2.0);
    try list.append(3.0);
    try list.append(4.0);
    try list.append(5.0);
    try list.append(6.0);

    // Call numpy.array()
    const arr_method = try np.getAttribute("array");
    defer arr_method.decref();

    return try py.call(root, py.PyObject, arr_method, .{list.obj}, .{});
}

/// Demonstrate array statistics
pub fn array_stats() !py.PyObject {
    const np = try py.numpy.getModule(@This());
    defer np.decref();

    // Create arange(1, 11)
    const arange_method = try np.getAttribute("arange");
    defer arange_method.decref();

    const arr = try py.call(root, py.PyObject, arange_method, .{ 1, 11 }, .{});
    defer arr.decref();

    // Get mean
    const mean_method = try arr.getAttribute("mean");
    defer mean_method.decref();
    const mean_val = try py.call(root, py.PyObject, mean_method, .{}, .{});

    // Create result dict
    const result = try py.PyDict(root).new();
    errdefer result.obj.decref();

    try result.setItem("array", arr);
    try result.setItem("mean", mean_val);

    // Increment refcount before returning
    arr.incref();
    mean_val.incref();

    return result.obj;
}

/// Check if NumPy is available
pub fn numpy_available() !bool {
    const np = py.numpy.getModule(@This()) catch return false;
    defer np.decref();
    return true;
}

comptime {
    py.rootmodule(root);
}
