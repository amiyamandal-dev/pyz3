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

//! NumPy C API Integration Example
//!
//! This example demonstrates using pyz3's NumPy C API bindings for
//! high-performance array operations with minimal Python overhead.

const std = @import("std");
const py = @import("pyz3");
const np_capi = py.numpy_capi;

const root = @This();

/// Compare performance: Python API vs C API for sum operation
pub fn compare_sum_performance(args: struct { arr: py.PyArray(root), iterations: usize }) !struct {
    python_api_ms: f64,
    c_api_ms: f64,
    speedup: f64,
} {
    var timer = try std.time.Timer.start();

    // Benchmark Python API
    var python_time: u64 = 0;
    for (0..args.iterations) |_| {
        timer.reset();
        _ = try args.arr.sum(f64);
        python_time += timer.read();
    }

    // Benchmark C API
    try np_capi.CAPI.initialize();
    const arr_ptr = @ptrCast(*np_capi.PyArrayObject, args.arr.obj.py);

    var capi_time: u64 = 0;
    for (0..args.iterations) |_| {
        timer.reset();

        const data = try np_capi.getData(f64, arr_ptr);
        var sum: f64 = 0.0;
        for (data) |val| {
            sum += val;
        }

        capi_time += timer.read();
    }

    const python_ms = @as(f64, @floatFromInt(python_time)) / 1_000_000.0;
    const capi_ms = @as(f64, @floatFromInt(capi_time)) / 1_000_000.0;

    return .{
        .python_api_ms = python_ms,
        .c_api_ms = capi_ms,
        .speedup = python_ms / capi_ms,
    };
}

/// Fast array sum using C API
pub fn sum_fast(args: struct { arr: py.PyArray(root) }) !f64 {
    try np_capi.CAPI.initialize();

    const arr_ptr = @ptrCast(*np_capi.PyArrayObject, args.arr.obj.py);
    const data = try np_capi.getData(f64, arr_ptr);

    var sum: f64 = 0.0;
    for (data) |val| {
        sum += val;
    }

    return sum;
}

/// Fast element-wise array multiplication using C API
pub fn multiply_fast(args: struct { arr: py.PyArray(root), scalar: f64 }) !py.PyArray(root) {
    try np_capi.CAPI.initialize();

    const arr_ptr = @ptrCast(*np_capi.PyArrayObject, args.arr.obj.py);
    const data = try np_capi.getDataMut(f64, arr_ptr);

    for (data) |*val| {
        val.* *= args.scalar;
    }

    return args.arr;
}

/// Compute dot product using C API (high performance)
pub fn dot_product_fast(args: struct { a: py.PyArray(root), b: py.PyArray(root) }) !f64 {
    try np_capi.CAPI.initialize();

    const a_ptr = @ptrCast(*np_capi.PyArrayObject, args.a.obj.py);
    const b_ptr = @ptrCast(*np_capi.PyArrayObject, args.b.obj.py);

    const data_a = try np_capi.getData(f64, a_ptr);
    const data_b = try np_capi.getData(f64, b_ptr);

    if (data_a.len != data_b.len) {
        return py.ValueError(root).raise("Arrays must have same length");
    }

    var result: f64 = 0.0;
    for (data_a, data_b) |a, b| {
        result += a * b;
    }

    return result;
}

/// Normalize array to unit length using C API
pub fn normalize_fast(args: struct { arr: py.PyArray(root) }) !py.PyArray(root) {
    try np_capi.CAPI.initialize();

    const arr_ptr = @ptrCast(*np_capi.PyArrayObject, args.arr.obj.py);
    const data = try np_capi.getDataMut(f64, arr_ptr);

    // Compute magnitude
    var magnitude: f64 = 0.0;
    for (data) |val| {
        magnitude += val * val;
    }
    magnitude = @sqrt(magnitude);

    if (magnitude < 1e-10) {
        return py.ValueError(root).raise("Cannot normalize zero-magnitude vector");
    }

    // Normalize
    for (data) |*val| {
        val.* /= magnitude;
    }

    return args.arr;
}

/// Create array from range using C API
pub fn arange_fast(args: struct { start: f64, stop: f64, step: f64 }) !py.PyArray(root) {
    try np_capi.CAPI.initialize();

    // Calculate size
    const size = @as(usize, @intFromFloat(@ceil((args.stop - args.start) / args.step)));

    // Create array
    const shape = [_]usize{size};
    const arr_ptr = try np_capi.zeros(f64, &shape, py.allocator);

    // Fill with values
    const data = try np_capi.getDataMut(f64, arr_ptr);
    var current = args.start;
    for (data) |*val| {
        val.* = current;
        current += args.step;
    }

    // Convert to PyArray for return
    const obj = @ptrCast(*py.ffi.PyObject, arr_ptr);
    return py.PyArray(root).from.unchecked(py.PyObject{ .py = obj });
}

/// Apply custom function to each element using C API
pub fn map_function_fast(
    args: struct { arr: py.PyArray(root) },
    func: fn (f64) f64,
) !py.PyArray(root) {
    try np_capi.CAPI.initialize();

    const arr_ptr = @ptrCast(*np_capi.PyArrayObject, args.arr.obj.py);
    const data = try np_capi.getDataMut(f64, arr_ptr);

    for (data) |*val| {
        val.* = func(val.*);
    }

    return args.arr;
}

/// Example custom function: square
fn square(x: f64) f64 {
    return x * x;
}

/// Apply square function using C API
pub fn square_fast(args: struct { arr: py.PyArray(root) }) !py.PyArray(root) {
    return try map_function_fast(args, square);
}

/// Get array statistics using C API (single pass)
pub fn statistics_fast(args: struct { arr: py.PyArray(root) }) !struct {
    count: usize,
    sum: f64,
    mean: f64,
    min: f64,
    max: f64,
    std: f64,
} {
    try np_capi.CAPI.initialize();

    const arr_ptr = @ptrCast(*np_capi.PyArrayObject, args.arr.obj.py);
    const data = try np_capi.getData(f64, arr_ptr);

    if (data.len == 0) {
        return py.ValueError(root).raise("Cannot compute statistics of empty array");
    }

    // First pass: min, max, sum
    var min_val = data[0];
    var max_val = data[0];
    var sum: f64 = 0.0;

    for (data) |val| {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }

    const mean = sum / @as(f64, @floatFromInt(data.len));

    // Second pass: standard deviation
    var variance: f64 = 0.0;
    for (data) |val| {
        const diff = val - mean;
        variance += diff * diff;
    }
    variance /= @as(f64, @floatFromInt(data.len));

    return .{
        .count = data.len,
        .sum = sum,
        .mean = mean,
        .min = min_val,
        .max = max_val,
        .std = @sqrt(variance),
    };
}

/// Matrix-vector multiplication using C API
pub fn matvec_fast(args: struct { matrix: py.PyArray(root), vector: py.PyArray(root) }) !py.PyArray(root) {
    try np_capi.CAPI.initialize();

    const mat_ptr = @ptrCast(*np_capi.PyArrayObject, args.matrix.obj.py);
    const vec_ptr = @ptrCast(*np_capi.PyArrayObject, args.vector.obj.py);

    // Get shapes
    const mat_shape = try np_capi.getShape(mat_ptr, py.allocator);
    defer py.allocator.free(mat_shape);
    const vec_shape = try np_capi.getShape(vec_ptr, py.allocator);
    defer py.allocator.free(vec_shape);

    // Verify dimensions
    if (mat_shape.len != 2) {
        return py.ValueError(root).raise("Matrix must be 2D");
    }
    if (vec_shape.len != 1) {
        return py.ValueError(root).raise("Vector must be 1D");
    }
    if (mat_shape[1] != vec_shape[0]) {
        return py.ValueError(root).raise("Incompatible dimensions for matrix-vector multiply");
    }

    const m = mat_shape[0];
    const n = mat_shape[1];

    // Get data
    const mat_data = try np_capi.getData(f64, mat_ptr);
    const vec_data = try np_capi.getData(f64, vec_ptr);

    // Create result vector
    const result_shape = [_]usize{m};
    const result_ptr = try np_capi.zeros(f64, &result_shape, py.allocator);
    const result_data = try np_capi.getDataMut(f64, result_ptr);

    // Perform multiplication
    for (0..m) |i| {
        var sum: f64 = 0.0;
        for (0..n) |j| {
            sum += mat_data[i * n + j] * vec_data[j];
        }
        result_data[i] = sum;
    }

    const obj = @ptrCast(*py.ffi.PyObject, result_ptr);
    return py.PyArray(root).from.unchecked(py.PyObject{ .py = obj });
}

/// Element-wise array addition using C API
pub fn add_arrays_fast(args: struct { a: py.PyArray(root), b: py.PyArray(root) }) !py.PyArray(root) {
    try np_capi.CAPI.initialize();

    const a_ptr = @ptrCast(*np_capi.PyArrayObject, args.a.obj.py);
    const b_ptr = @ptrCast(*np_capi.PyArrayObject, args.b.obj.py);

    const data_a = try np_capi.getData(f64, a_ptr);
    const data_b = try np_capi.getData(f64, b_ptr);

    if (data_a.len != data_b.len) {
        return py.ValueError(root).raise("Arrays must have same size");
    }

    // Create result array
    const result_shape = [_]usize{data_a.len};
    const result_ptr = try np_capi.zeros(f64, &result_shape, py.allocator);
    const result_data = try np_capi.getDataMut(f64, result_ptr);

    for (data_a, data_b, result_data) |a, b, *r| {
        r.* = a + b;
    }

    const obj = @ptrCast(*py.ffi.PyObject, result_ptr);
    return py.PyArray(root).from.unchecked(py.PyObject{ .py = obj });
}

comptime {
    py.rootmodule(root);
}
