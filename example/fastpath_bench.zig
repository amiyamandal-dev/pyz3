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

/// Example module to benchmark fast path optimization for primitive types
const std = @import("std");
const py = @import("pyz3");

/// Test i64 fast path
pub fn return_i64(args: struct { value: i64 }) i64 {
    return args.value * 2;
}

/// Test f64 fast path
pub fn return_f64(args: struct { value: f64 }) f64 {
    return args.value * 2.0;
}

/// Test bool fast path
pub fn return_bool(args: struct { value: bool }) bool {
    return !args.value;
}

/// Test string fast path - returns length of input string
pub fn return_string(args: struct { value: []const u8 }) u64 {
    return args.value.len;
}

/// Test mixed types
pub fn mixed_types(args: struct {
    int_val: i64,
    float_val: f64,
    bool_val: bool,
    str_val: []const u8
}) !py.PyDict(@This()) {
    const dict = try py.PyDict(@This()).new();

    const int_obj = try py.create(@This(), args.int_val * 2);
    try dict.setItem(try py.PyString.create("int_result"), int_obj);

    const float_obj = try py.create(@This(), args.float_val * 2.0);
    try dict.setItem(try py.PyString.create("float_result"), float_obj);

    const bool_obj = try py.create(@This(), !args.bool_val);
    try dict.setItem(try py.PyString.create("bool_result"), bool_obj);

    const str_obj = try py.create(@This(), args.str_val);
    try dict.setItem(try py.PyString.create("str_result"), str_obj);

    return dict;
}

/// Benchmark primitive conversions
pub fn benchmark_primitives(args: struct { iterations: i64 }) !i64 {
    const iterations: usize = @intCast(args.iterations);
    var sum: i64 = 0;

    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        // These will use fast paths
        const int_obj = try py.create(@This(), @as(i64, @intCast(i)));
        const int_val = try py.as(@This(), i64, int_obj);
        sum += int_val;
        int_obj.decref();
    }

    return sum;
}

/// Test optional types
pub fn optional_int(args: struct { value: ?i64 }) ?i64 {
    if (args.value) |v| {
        return v * 2;
    }
    return null;
}

/// Test error unions
pub fn error_union_int(args: struct { value: i64, should_error: bool }) !i64 {
    if (args.should_error) {
        return error.TestError;
    }
    return args.value * 2;
}

comptime {
    py.rootmodule(@This());
}

test "i64 fast path" {
    py.initialize();
    defer py.finalize();

    const result = return_i64(.{ .value = 21 });
    try std.testing.expectEqual(@as(i64, 42), result);
}

test "f64 fast path" {
    py.initialize();
    defer py.finalize();

    const result = return_f64(.{ .value = 21.5 });
    try std.testing.expectEqual(@as(f64, 43.0), result);
}

test "bool fast path" {
    py.initialize();
    defer py.finalize();

    const result = return_bool(.{ .value = true });
    try std.testing.expectEqual(false, result);
}

test "string fast path" {
    py.initialize();
    defer py.finalize();

    const result = return_string(.{ .value = "hello" });
    try std.testing.expectEqual(@as(u64, 5), result); // Length of "hello"
}

test "optional int" {
    py.initialize();
    defer py.finalize();

    const result1 = optional_int(.{ .value = 21 });
    try std.testing.expectEqual(@as(?i64, 42), result1);

    const result2 = optional_int(.{ .value = null });
    try std.testing.expectEqual(@as(?i64, null), result2);
}
