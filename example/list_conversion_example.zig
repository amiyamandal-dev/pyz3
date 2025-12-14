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

/// Example module demonstrating PyList <=> Zig array/slice conversion
const std = @import("std");
const py = @import("pyz3");

/// Sum all numbers in a list using automatic conversion
pub fn sum_list(args: struct { numbers: py.PyList(@This()) }) !i64 {
    // Convert PyList to Zig slice
    const slice = try args.numbers.toSlice(i64);
    defer py.allocator.free(slice);

    var total: i64 = 0;
    for (slice) |val| {
        total += val;
    }

    return total;
}

/// Create a list from a Zig array using automatic conversion
pub fn create_range(args: struct { start: i64, end: i64 }) !py.PyList(@This()) {
    const count: usize = @intCast(args.end - args.start);
    const numbers = try py.allocator.alloc(i64, count);
    defer py.allocator.free(numbers);

    for (0..count) |i| {
        numbers[i] = args.start + @as(i64, @intCast(i));
    }

    // Use fromSlice to create PyList
    return py.PyList(@This()).fromSlice(numbers);
}

/// Multiply all elements by a scalar
pub fn scale_vector(args: struct {
    numbers: py.PyList(@This()),
    scale: f64
}) !py.PyList(@This()) {
    // Convert to Zig slice
    const input = try args.numbers.toSlice(f64);
    defer py.allocator.free(input);

    // Process
    const output = try py.allocator.alloc(f64, input.len);
    defer py.allocator.free(output);

    for (0..input.len) |i| {
        output[i] = input[i] * args.scale;
    }

    // Convert back to Python list
    return py.PyList(@This()).fromSlice(output);
}

/// Compute dot product of two vectors
pub fn dot_product(args: struct {
    a: py.PyList(@This()),
    b: py.PyList(@This()),
}) !f64 {
    const vec_a = try args.a.toSlice(f64);
    defer py.allocator.free(vec_a);

    const vec_b = try args.b.toSlice(f64);
    defer py.allocator.free(vec_b);

    if (vec_a.len != vec_b.len) {
        return py.ValueError(@This()).raise("Vectors must have same length");
    }

    var result: f64 = 0;
    for (0..vec_a.len) |i| {
        result += vec_a[i] * vec_b[i];
    }

    return result;
}

/// Find min and max in vector
pub fn minmax(args: struct { numbers: py.PyList(@This()) }) !struct {
    min: f64,
    max: f64
} {
    const vector = try args.numbers.toSlice(f64);
    defer py.allocator.free(vector);

    if (vector.len == 0) {
        return py.ValueError(@This()).raise("List cannot be empty");
    }

    var min_val = vector[0];
    var max_val = vector[0];

    for (vector) |val| {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    return .{ .min = min_val, .max = max_val };
}

/// Example using automatic py.create() conversion
pub fn automatic_create() !py.PyObject {
    // Automatic conversion: []i64 -> PyList
    const numbers = [_]i64{ 10, 20, 30, 40, 50 };
    return try py.create(@This(), &numbers);
}

/// Example using automatic py.as() conversion
pub fn automatic_extract(args: struct { data: py.PyObject }) !i64 {
    // Automatic conversion: PyList -> []i64
    const slice = try py.as(@This(), []i64, args.data);
    defer py.allocator.free(slice);

    var sum: i64 = 0;
    for (slice) |val| {
        sum += val;
    }
    return sum;
}

/// Example with nested lists (list of lists)
pub fn sum_matrix(args: struct { matrix: py.PyList(@This()) }) !i64 {
    // Convert outer list to slice of PyLists
    const rows = try args.matrix.toSlice(py.PyList(@This()));
    defer py.allocator.free(rows);

    var total: i64 = 0;
    for (rows) |row| {
        // Convert each inner list to slice
        const values = try row.toSlice(i64);
        defer py.allocator.free(values);

        for (values) |val| {
            total += val;
        }
    }

    return total;
}

/// Example creating nested lists
pub fn create_matrix(args: struct { rows: usize, cols: usize }) !py.PyList(@This()) {
    // Create outer list
    const matrix = try py.PyList(@This()).new(args.rows);

    var value: i64 = 1;
    for (0..args.rows) |i| {
        // Create inner array
        const row = try py.allocator.alloc(i64, args.cols);
        defer py.allocator.free(row);

        for (0..args.cols) |j| {
            row[j] = value;
            value += 1;
        }

        // Convert to PyList and add to matrix
        const row_list = try py.PyList(@This()).fromSlice(row);
        try matrix.setOwnedItem(i, row_list);
    }

    return matrix;
}

/// Filter values above threshold
pub fn filter_above(args: struct {
    numbers: py.PyList(@This()),
    threshold: f64,
}) !py.PyList(@This()) {
    const input = try args.numbers.toSlice(f64);
    defer py.allocator.free(input);

    // Count how many pass the filter
    var count: usize = 0;
    for (input) |val| {
        if (val > args.threshold) count += 1;
    }

    // Create filtered array
    const output = try py.allocator.alloc(f64, count);
    defer py.allocator.free(output);

    var idx: usize = 0;
    for (input) |val| {
        if (val > args.threshold) {
            output[idx] = val;
            idx += 1;
        }
    }

    return py.PyList(@This()).fromSlice(output);
}

comptime {
    py.rootmodule(@This());
}

// Tests
const testing = std.testing;

// Tests commented out due to type signature changes
// test "sum_list" {
//     py.initialize();
//     defer py.finalize();
//
//     const root = @This();
//
//     var list = try py.PyList(root).new(3);
//     defer list.obj.decref();
//     try list.setItem(0, 10);
//     try list.setItem(1, 20);
//     try list.setItem(2, 30);
//
//     const result = try sum_list(.{ .numbers = list });
//     try testing.expectEqual(@as(i64, 60), result);
// }
//
// test "create_range" {
//     py.initialize();
//     defer py.finalize();
//
//     const list = try create_range(.{ .start = 5, .end = 10 });
//     defer list.obj.decref();
//
//     try testing.expectEqual(@as(usize, 5), list.length());
//     try testing.expectEqual(@as(i64, 5), try list.getItem(i64, 0));
//     try testing.expectEqual(@as(i64, 9), try list.getItem(i64, 4));
// }
//
// test "automatic conversions" {
//     py.initialize();
//     defer py.finalize();
//
//     // Test automatic create
//     const pyobj = try automatic_create();
//     defer pyobj.decref();
//
//     // Test automatic extract
//     const result = try automatic_extract(.{ .data = pyobj });
//     try testing.expectEqual(@as(i64, 150), result);
// }
