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
const py = @import("pyz3");

const root = @This();

/// Create a new NumPy array filled with zeros
pub fn create_zeros(args: struct { size: usize }) !py.PyArray(root) {
    return try py.PyArray(root).zeros(f64, &[_]usize{args.size});
}

/// Create a new NumPy array filled with ones
pub fn create_ones(args: struct { rows: usize, cols: usize }) !py.PyArray(root) {
    return try py.PyArray(root).ones(f64, &[_]usize{ args.rows, args.cols });
}

/// Create a new NumPy array from a Zig slice
pub fn from_slice(args: struct { values: []const f64 }) !py.PyArray(root) {
    return try py.PyArray(root).fromSlice(f64, args.values);
}

/// Double all elements in a NumPy array (in-place, zero-copy)
pub fn double_array(args: struct { arr: py.PyArray(root) }) !py.PyArray(root) {
    // Get mutable zero-copy access to the array data
    const data = try args.arr.asSliceMut(f64);

    // Modify in-place
    for (data) |*val| {
        val.* *= 2.0;
    }

    return args.arr;
}

/// Sum all elements in a NumPy array (zero-copy read)
pub fn sum_array(args: struct { arr: py.PyArray(root) }) !f64 {
    // Get read-only zero-copy access
    const data = try args.arr.asSlice(f64);

    var sum: f64 = 0.0;
    for (data) |val| {
        sum += val;
    }

    return sum;
}

/// Calculate mean using NumPy's built-in method
pub fn calculate_mean(args: struct { arr: py.PyArray(root) }) !f64 {
    return try args.arr.mean(f64);
}

/// Get array statistics
pub fn array_stats(args: struct { arr: py.PyArray(root) }) !struct { min: f64, max: f64, mean: f64, sum: f64 } {
    return .{
        .min = try args.arr.min(f64),
        .max = try args.arr.max(f64),
        .mean = try args.arr.mean(f64),
        .sum = try args.arr.sum(f64),
    };
}

/// Reshape an array
pub fn reshape_array(args: struct { arr: py.PyArray(root), rows: usize, cols: usize }) !py.PyArray(root) {
    return try args.arr.reshape(&[_]usize{ args.rows, args.cols });
}

/// Flatten a multi-dimensional array
pub fn flatten_array(args: struct { arr: py.PyArray(root) }) !py.PyArray(root) {
    return try args.arr.flatten();
}

/// Transpose an array
pub fn transpose_array(args: struct { arr: py.PyArray(root) }) !py.PyArray(root) {
    return try args.arr.transpose();
}

/// Get array metadata
pub fn array_info(args: struct { arr: py.PyArray(root) }) !struct { ndim: usize, size: usize, dtype: []const u8 } {
    const shape = try args.arr.shape();
    const dtype = try args.arr.dtype();

    return .{
        .ndim = try args.arr.ndim(),
        .size = try args.arr.size(),
        .dtype = try dtype.str(),
    };
}

/// Perform element-wise operations on two arrays
pub fn add_arrays(args: struct { a: py.PyArray(root), b: py.PyArray(root) }) !py.PyArray(root) {
    // Verify same size
    const size_a = try args.a.size();
    const size_b = try args.b.size();

    if (size_a != size_b) {
        return py.ValueError(root).raise("Arrays must have the same size");
    }

    // Get data from both arrays
    const data_a = try args.a.asSlice(f64);
    const data_b = try args.b.asSlice(f64);

    // Create result array
    const result = try py.PyArray(root).zeros(f64, &[_]usize{size_a});
    const result_data = try result.asSliceMut(f64);

    // Add element-wise
    for (data_a, data_b, result_data) |a, b, *r| {
        r.* = a + b;
    }

    return result;
}

/// Example class that holds a NumPy array
pub const ArrayProcessor = py.class(struct {
    pub const __doc__ = "A class for processing NumPy arrays";
    const Self = @This();

    arr: py.PyArray(root),

    pub fn __init__(self: *Self, args: struct { arr: py.PyArray(root) }) !void {
        self.* = .{ .arr = args.arr };
    }

    pub fn get_array(self: *const Self) py.PyArray(root) {
        return self.arr;
    }

    pub fn double(self: *const Self) !py.PyArray(root) {
        const data = try self.arr.asSliceMut(f64);
        for (data) |*val| {
            val.* *= 2.0;
        }
        return self.arr;
    }

    pub fn sum(self: *const Self) !f64 {
        return try self.arr.sum(f64);
    }

    pub fn mean(self: *const Self) !f64 {
        return try self.arr.mean(f64);
    }
});

comptime {
    py.rootmodule(root);
}
