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

/// Example module demonstrating SIMD operations with pyz3
const std = @import("std");
const py = @import("pyz3");
const simd = @import("pyz3").simd;

/// Add two 4-element float vectors
pub fn vec4_add(args: struct {
    a: py.PyObject,
    b: py.PyObject,
}) !py.PyList(@This()) {
    const vec_a = try simd.fromPython(f32, 4, args.a);
    const vec_b = try simd.fromPython(f32, 4, args.b);

    const result = simd.SimdOps.add(f32, 4, vec_a, vec_b);

    return simd.toPython(@This(), f32, 4, result);
}

/// Compute dot product of two vectors
pub fn vec4_dot(args: struct {
    a: py.PyObject,
    b: py.PyObject,
}) !f32 {
    const vec_a = try simd.fromPython(f32, 4, args.a);
    const vec_b = try simd.fromPython(f32, 4, args.b);

    return simd.SimdOps.dot(f32, 4, vec_a, vec_b);
}

/// Scale a vector by a scalar
pub fn vec4_scale(args: struct {
    vec: py.PyObject,
    scalar: f32,
}) !py.PyList(@This()) {
    const vec = try simd.fromPython(f32, 4, args.vec);
    const result = simd.SimdOps.scale(f32, 4, vec, args.scalar);

    return simd.toPython(@This(), f32, 4, result);
}

/// Sum all elements in a vector
pub fn vec4_sum(args: struct { vec: py.PyObject }) !f32 {
    const vec = try simd.fromPython(f32, 4, args.vec);
    return simd.SimdOps.sum(f32, 4, vec);
}

/// Find minimum element in vector
pub fn vec4_min(args: struct { vec: py.PyObject }) !f32 {
    const vec = try simd.fromPython(f32, 4, args.vec);
    return simd.SimdOps.min(f32, 4, vec);
}

/// Find maximum element in vector
pub fn vec4_max(args: struct { vec: py.PyObject }) !f32 {
    const vec = try simd.fromPython(f32, 4, args.vec);
    return simd.SimdOps.max(f32, 4, vec);
}

/// Batch add arrays using SIMD
pub fn batch_add(args: struct {
    a: py.PyObject,
    b: py.PyObject,
}) !py.PyList(@This()) {
    // Convert PyObject to PyList
    const list_a = py.PyList(@This()){ .obj = args.a };
    const list_b = py.PyList(@This()){ .obj = args.b };

    const len_a = list_a.length();
    const len_b = list_b.length();

    if (len_a != len_b) {
        return py.ValueError(@This()).raise("Arrays must have same length");
    }

    // Allocate result arrays
    const allocator = py.allocator;
    const arr_a = try allocator.alloc(f32, len_a);
    defer allocator.free(arr_a);

    const arr_b = try allocator.alloc(f32, len_b);
    defer allocator.free(arr_b);

    const result = try allocator.alloc(f32, len_a);
    defer allocator.free(result);

    // Convert Python lists to arrays
    for (0..len_a) |i| {
        const item_a = try list_a.getItem(py.PyObject, @intCast(i));
        arr_a[i] = try py.as(@This(), f32, item_a);

        const item_b = try list_b.getItem(py.PyObject, @intCast(i));
        arr_b[i] = try py.as(@This(), f32, item_b);
    }

    // Perform SIMD batch add
    // Create a wrapper function since SimdOps.add is generic
    const AddF32x4 = struct {
        fn op(a: simd.SimdVec(f32, 4), b: simd.SimdVec(f32, 4)) simd.SimdVec(f32, 4) {
            return simd.SimdOps.add(f32, 4, a, b);
        }
    };
    simd.batchOp(f32, 4, AddF32x4.op, arr_a, arr_b, result);

    // Convert result to Python list
    const py_result = try py.PyList(@This()).new(0);
    for (result) |val| {
        const item = try py.create(@This(), val);
        try py_result.append(item);
        item.decref();
    }

    return py_result;
}

/// Compute distance between two 4D points
pub fn vec4_distance(args: struct {
    a: py.PyObject,
    b: py.PyObject,
}) !f32 {
    const vec_a = try simd.fromPython(f32, 4, args.a);
    const vec_b = try simd.fromPython(f32, 4, args.b);

    // Compute difference
    const diff = simd.SimdOps.sub(f32, 4, vec_a, vec_b);

    // Compute squared distance (dot product with itself)
    const squared_dist = simd.SimdOps.dot(f32, 4, diff, diff);

    // Return square root
    return @sqrt(squared_dist);
}

comptime {
    py.rootmodule(@This());
}

test "vec4 add" {
    py.initialize();
    defer py.finalize();

    const a = try py.PyList(@This()).new(0);
    defer a.obj.decref();

    try a.append((try py.PyFloat.create(1.0)).obj);
    try a.append((try py.PyFloat.create(2.0)).obj);
    try a.append((try py.PyFloat.create(3.0)).obj);
    try a.append((try py.PyFloat.create(4.0)).obj);

    const b = try py.PyList(@This()).new(0);
    defer b.obj.decref();

    try b.append((try py.PyFloat.create(5.0)).obj);
    try b.append((try py.PyFloat.create(6.0)).obj);
    try b.append((try py.PyFloat.create(7.0)).obj);
    try b.append((try py.PyFloat.create(8.0)).obj);

    const result = try vec4_add(.{ .a = a.obj, .b = b.obj });
    defer result.obj.decref();

    try std.testing.expectEqual(@as(usize, 4), result.length());
}

test "vec4 dot product" {
    py.initialize();
    defer py.finalize();

    const a = try py.PyList(@This()).new(0);
    defer a.obj.decref();

    try a.append((try py.PyFloat.create(1.0)).obj);
    try a.append((try py.PyFloat.create(2.0)).obj);
    try a.append((try py.PyFloat.create(3.0)).obj);
    try a.append((try py.PyFloat.create(4.0)).obj);

    const b = try py.PyList(@This()).new(0);
    defer b.obj.decref();

    try b.append((try py.PyFloat.create(5.0)).obj);
    try b.append((try py.PyFloat.create(6.0)).obj);
    try b.append((try py.PyFloat.create(7.0)).obj);
    try b.append((try py.PyFloat.create(8.0)).obj);

    const result = try vec4_dot(.{ .a = a.obj, .b = b.obj });

    // 1*5 + 2*6 + 3*7 + 4*8 = 70
    try std.testing.expectEqual(@as(f32, 70.0), result);
}
