// Example demonstrating NumPy C API integration in pyz3
// Shows zero-copy access to NumPy arrays from Zig

const std = @import("std");
const py = @import("pyz3");

const root = @This();

/// Create a NumPy array from Zig slice using PyArray
pub fn create_array() !py.PyObject {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const arr = try py.PyArray(root).fromSlice(f64, &data);
    return arr.obj;
}

/// Double all elements in a NumPy array using zero-copy C API access
pub fn double_array(args: struct { arr: py.PyObject }) !py.PyObject {
    // Wrap the incoming PyObject as a PyArray
    const arr = try py.PyArray(root).from.checked(args.arr);

    // Get mutable slice - this is ZERO-COPY direct memory access!
    const data = try arr.asSliceMut(f64);

    // Double each element in-place
    for (data) |*val| {
        val.* *= 2.0;
    }

    // Return the same array (modified in place)
    arr.incref();
    return arr.obj;
}

/// Sum array elements using SIMD-optimized loop with zero-copy access
pub fn fast_sum(args: struct { arr: py.PyObject }) !f64 {
    const arr = try py.PyArray(root).from.checked(args.arr);

    // Zero-copy access to array data
    const data = try arr.asSlice(f64);

    // Simple sum (could use SIMD here)
    var total: f64 = 0.0;
    for (data) |val| {
        total += val;
    }
    return total;
}

/// Get array info using C API direct access
pub fn array_info(args: struct { arr: py.PyObject }) !py.PyObject {
    const arr = try py.PyArray(root).from.checked(args.arr);

    // All these are direct C API calls - no Python overhead
    const ndim_val = arr.ndim();
    const size_val = arr.size();
    const itemsize_val = arr.itemsize();
    const nbytes_val = arr.nbytes();
    const is_c_contiguous = arr.isCContiguous();
    const is_writeable = arr.isWriteable();

    // Get dtype name
    const dtype_name = if (arr.dtype()) |dt| dt.name() else "unknown";

    // Create result dict
    const result = try py.PyDict(root).new();
    errdefer result.obj.decref();

    try result.setItem("ndim", ndim_val);
    try result.setItem("size", size_val);
    try result.setItem("itemsize", itemsize_val);
    try result.setItem("nbytes", nbytes_val);
    try result.setItem("c_contiguous", is_c_contiguous);
    try result.setItem("writeable", is_writeable);
    try result.setItem("dtype", dtype_name);

    return result.obj;
}

/// Get array shape as a list (direct C API access to dims)
pub fn get_shape(args: struct { arr: py.PyObject }) !py.PyObject {
    const arr = try py.PyArray(root).from.checked(args.arr);

    // Direct access to shape array via C API
    const shape_slice = arr.shape();

    const result = try py.PyList(root).new(0);
    for (shape_slice) |dim| {
        try result.append(@as(i64, @intCast(dim)));
    }
    return result.obj;
}

/// Get array strides as a list (direct C API access)
pub fn get_strides(args: struct { arr: py.PyObject }) !py.PyObject {
    const arr = try py.PyArray(root).from.checked(args.arr);

    const strides_slice = arr.strides();

    const result = try py.PyList(root).new(0);
    for (strides_slice) |s| {
        try result.append(@as(i64, @intCast(s)));
    }
    return result.obj;
}

/// Get array statistics using PyArray methods
pub fn array_stats(args: struct { arr: py.PyObject }) !py.PyObject {
    const arr = try py.PyArray(root).from.checked(args.arr);

    const sum_val = try arr.sum(f64);
    const mean_val = try arr.mean();
    const min_val = try arr.min(f64);
    const max_val = try arr.max(f64);

    const result = try py.PyDict(root).new();
    errdefer result.obj.decref();

    try result.setItem("sum", sum_val);
    try result.setItem("mean", mean_val);
    try result.setItem("min", min_val);
    try result.setItem("max", max_val);

    return result.obj;
}

/// Create a zeros array with specified shape
pub fn create_zeros(args: struct { rows: usize, cols: usize }) !py.PyObject {
    const shape = [_]usize{ args.rows, args.cols };
    const arr = try py.PyArray(root).zeros(f64, &shape);
    return arr.obj;
}

/// Create a ones array with specified shape
pub fn create_ones(args: struct { rows: usize, cols: usize }) !py.PyObject {
    const shape = [_]usize{ args.rows, args.cols };
    const arr = try py.PyArray(root).ones(f64, &shape);
    return arr.obj;
}

/// Multiply two arrays element-wise using zero-copy access
pub fn elementwise_multiply(args: struct { a: py.PyObject, b: py.PyObject }) !py.PyObject {
    const arr_a = try py.PyArray(root).from.checked(args.a);
    const arr_b = try py.PyArray(root).from.checked(args.b);

    // Verify both arrays have float64 dtype
    const dtype_a = arr_a.dtype() orelse {
        return py.TypeError(root).raise("First array has unsupported dtype, expected float64");
    };
    const dtype_b = arr_b.dtype() orelse {
        return py.TypeError(root).raise("Second array has unsupported dtype, expected float64");
    };

    if (dtype_a != .float64) {
        return py.TypeError(root).raise("First array must be float64");
    }
    if (dtype_b != .float64) {
        return py.TypeError(root).raise("Second array must be float64");
    }

    // Get read-only access to input arrays
    const data_a = try arr_a.asSlice(f64);
    const data_b = try arr_b.asSlice(f64);

    if (data_a.len != data_b.len) {
        return py.ValueError(root).raise("Arrays must have the same size");
    }

    // Handle empty arrays
    if (data_a.len == 0) {
        const shape = [_]usize{0};
        const result = try py.PyArray(root).empty(f64, &shape);
        return result.obj;
    }

    // Create output array
    const shape = [_]usize{data_a.len};
    const result = try py.PyArray(root).empty(f64, &shape);

    // Get mutable access to output
    const data_out = try result.asSliceMut(f64);

    // Perform element-wise multiplication
    for (data_a, data_b, data_out) |a, b, *out| {
        out.* = a * b;
    }

    return result.obj;
}

/// Check if NumPy is available
pub fn numpy_available() !bool {
    const np = py.import(root, "numpy") catch return false;
    defer np.decref();
    return true;
}

comptime {
    py.rootmodule(root);
}
