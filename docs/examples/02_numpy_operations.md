# Example 2: NumPy Array Operations

High-performance NumPy operations using PyZ3.

## src/numpy_ops.zig

```zig
const std = @import("std");
const py = @import("pyz3");

/// Add two NumPy arrays element-wise (fast version)
pub fn add_arrays(a: py.PyObject, b: py.PyObject) !py.PyObject {
    const np = try py.PyModule.import("numpy");
    const add_func = try np.getAttr("add");
    return try add_func.call(.{a, b});
}

/// Multiply array by scalar using SIMD
pub fn multiply_scalar(array: py.PyObject, scalar: f64) !py.PyObject {
    // Get array info
    const np = try py.PyModule.import("numpy");
    const ascontiguousarray = try np.getAttr("ascontiguousarray");
    const arr = try ascontiguousarray.call(.{array});

    const shape = try arr.getAttr("shape");
    const size_obj = try arr.getAttr("size");
    const size = try size_obj.asLong();
    const n: usize = @intCast(size);

    // Get data pointer
    const ctypes = try arr.getAttr("ctypes");
    const data_ptr_obj = try ctypes.getAttr("data");
    const data_ptr_int = try data_ptr_obj.asLong();
    const data: [*]f64 = @ptrFromInt(@as(usize, @intCast(data_ptr_int)));

    // Create result array
    const empty = try np.getAttr("empty");
    const result = try empty.call(.{shape});
    const result_ctypes = try result.getAttr("ctypes");
    const result_ptr_obj = try result_ctypes.getAttr("data");
    const result_ptr_int = try result_ptr_obj.asLong();
    const result_data: [*]f64 = @ptrFromInt(@as(usize, @intCast(result_ptr_int)));

    // SIMD multiplication (4 elements at a time)
    const Vec = @Vector(4, f64);
    const scalar_vec = @as(Vec, @splat(scalar));

    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        const input_vec: Vec = data[i..][0..4].*;
        const output_vec = input_vec * scalar_vec;
        result_data[i..][0..4].* = output_vec;
    }

    // Handle remaining elements
    while (i < n) : (i += 1) {
        result_data[i] = data[i] * scalar;
    }

    return result;
}

/// Calculate mean of array (faster than NumPy for small arrays)
pub fn fast_mean(array: py.PyObject) !f64 {
    const np = try py.PyModule.import("numpy");
    const ascontiguousarray = try np.getAttr("ascontiguousarray");
    const arr = try ascontiguousarray.call(.{array});

    const size_obj = try arr.getAttr("size");
    const size = try size_obj.asLong();
    const n: usize = @intCast(size);

    const ctypes = try arr.getAttr("ctypes");
    const data_ptr_obj = try ctypes.getAttr("data");
    const data_ptr_int = try data_ptr_obj.asLong();
    const data: [*]f64 = @ptrFromInt(@as(usize, @intCast(data_ptr_int)));

    // SIMD sum
    const Vec = @Vector(4, f64);
    var sum_vec = Vec{0, 0, 0, 0};

    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        const vec: Vec = data[i..][0..4].*;
        sum_vec += vec;
    }

    var total: f64 = 0;
    inline for (0..4) |j| {
        total += sum_vec[j];
    }

    while (i < n) : (i += 1) {
        total += data[i];
    }

    return total / @as(f64, @floatFromInt(n));
}

/// Matrix multiplication (optimized for cache)
pub fn matmul(a: py.PyObject, b: py.PyObject) !py.PyObject {
    const np = try py.PyModule.import("numpy");

    // Ensure C-contiguous arrays
    const ascontiguousarray = try np.getAttr("ascontiguousarray");
    const a_contig = try ascontiguousarray.call(.{a});
    const b_contig = try ascontiguousarray.call(.{b});

    // Get shapes
    const a_shape = try a_contig.getAttr("shape");
    const b_shape = try b_contig.getAttr("shape");

    const m = try (try a_shape.getItem(0)).asLong();
    const k = try (try a_shape.getItem(1)).asLong();
    const n = try (try b_shape.getItem(1)).asLong();

    // Get data pointers
    const a_ctypes = try a_contig.getAttr("ctypes");
    const a_ptr_int = try (try a_ctypes.getAttr("data")).asLong();
    const a_data: [*]f64 = @ptrFromInt(@as(usize, @intCast(a_ptr_int)));

    const b_ctypes = try b_contig.getAttr("ctypes");
    const b_ptr_int = try (try b_ctypes.getAttr("data")).asLong();
    const b_data: [*]f64 = @ptrFromInt(@as(usize, @intCast(b_ptr_int)));

    // Create result
    const zeros = try np.getAttr("zeros");
    const result = try zeros.call(.{.{m, n}});
    const r_ctypes = try result.getAttr("ctypes");
    const r_ptr_int = try (try r_ctypes.getAttr("data")).asLong();
    const r_data: [*]f64 = @ptrFromInt(@as(usize, @intCast(r_ptr_int)));

    // Cache-friendly matrix multiplication
    const m_usize: usize = @intCast(m);
    const k_usize: usize = @intCast(k);
    const n_usize: usize = @intCast(n);

    for (0..m_usize) |i| {
        for (0..n_usize) |j| {
            var sum: f64 = 0;
            for (0..k_usize) |l| {
                sum += a_data[i * k_usize + l] * b_data[l * n_usize + j];
            }
            r_data[i * n_usize + j] = sum;
        }
    }

    return result;
}

comptime {
    py.rootmodule(@This());
}
```

## test_numpy_ops.py

```python
import numpy as np
import pytest
from numpy_ops import add_arrays, multiply_scalar, fast_mean, matmul

def test_add_arrays():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    result = add_arrays(a, b)
    expected = np.array([5.0, 7.0, 9.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_multiply_scalar():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    result = multiply_scalar(arr, 2.5)
    expected = arr * 2.5
    np.testing.assert_array_almost_equal(result, expected)

def test_multiply_scalar_large():
    """Test SIMD path with large array"""
    arr = np.random.rand(10000)
    result = multiply_scalar(arr, 3.14)
    expected = arr * 3.14
    np.testing.assert_array_almost_equal(result, expected)

def test_fast_mean():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fast_mean(arr)
    expected = np.mean(arr)
    assert abs(result - expected) < 1e-10

def test_fast_mean_large():
    arr = np.random.rand(100000)
    result = fast_mean(arr)
    expected = np.mean(arr)
    assert abs(result - expected) < 1e-6

def test_matmul():
    a = np.random.rand(100, 50)
    b = np.random.rand(50, 75)
    result = matmul(a, b)
    expected = np.matmul(a, b)
    np.testing.assert_array_almost_equal(result, expected, decimal=10)

def test_matmul_small():
    a = np.array([[1, 2], [3, 4]], dtype=float)
    b = np.array([[5, 6], [7, 8]], dtype=float)
    result = matmul(a, b)
    expected = np.matmul(a, b)
    np.testing.assert_array_almost_equal(result, expected)

# Benchmark
if __name__ == "__main__":
    import time

    print("Benchmarking NumPy operations...")

    # Test multiply_scalar
    arr = np.random.rand(1000000)

    start = time.time()
    for _ in range(100):
        _ = arr * 2.5
    numpy_time = time.time() - start

    start = time.time()
    for _ in range(100):
        _ = multiply_scalar(arr, 2.5)
    zig_time = time.time() - start

    print(f"multiply_scalar: NumPy={numpy_time:.4f}s, Zig={zig_time:.4f}s, Speedup={numpy_time/zig_time:.2f}x")

    # Test matmul
    a = np.random.rand(500, 500)
    b = np.random.rand(500, 500)

    start = time.time()
    _ = np.matmul(a, b)
    numpy_time = time.time() - start

    start = time.time()
    _ = matmul(a, b)
    zig_time = time.time() - start

    print(f"matmul: NumPy={numpy_time:.4f}s, Zig={zig_time:.4f}s, Speedup={numpy_time/zig_time:.2f}x")
```

## pyproject.toml

```toml
[build-system]
requires = ["pyz3>=0.8.0"]
build-backend = "pyz3.build"

[project]
name = "numpy-ops-ext"
version = "0.1.0"
dependencies = ["numpy>=1.20.0"]

[tool.pyz3.ext-module.numpy_ops]
root = "src/numpy_ops.zig"
```
