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
const py = @import("../pyz3.zig");
const PyObjectMixin = @import("./obj.zig").PyObjectMixin;
const ffi = py.ffi;
const PyError = @import("../errors.zig").PyError;

/// Wrapper for Python collections.defaultdict
pub fn PyDefaultDict(comptime root: type) type {
    return extern struct {
        obj: py.PyObject,

        const Self = @This();
        pub const from = PyObjectMixin("defaultdict", "PyDefaultDict", Self);

        /// Create a new defaultdict.
        pub fn new(default_factory: py.PyObject) !Self {
            const collections = try py.import(root, "collections");
            defer collections.decref();

            const defaultdict_type = try collections.get("defaultdict");
            defer defaultdict_type.decref();

            const dict_obj = try py.call(root, py.PyObject, defaultdict_type, .{default_factory}, .{});
            return .{ .obj = dict_obj };
        }

        /// Get the default_factory.
        pub fn defaultFactory(self: Self) !py.PyObject {
            return self.obj.get("default_factory");
        }

        /// Delegate to PyDict methods
        pub fn asDict(self: Self) py.PyDict(root) {
            return .{ .obj = self.obj };
        }
    };
}

test "PyDefaultDict" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    // Get the list type to use as the default factory
    const builtins = try py.import(root, "builtins");
    defer builtins.decref();
    const list_type = try builtins.get("list");
    defer list_type.decref();

    // Create a defaultdict
    const dd = try py.PyDefaultDict(root).new(list_type);
    defer dd.obj.decref();

    const dd_dict = dd.asDict();

    // Access a missing key, should create a new list
    const missing_key = try py.PyString.create("missing");
    defer missing_key.obj.decref();

    const default_val = try dd_dict.getItem(py.PyObject, missing_key);
    try std.testing.expect(default_val != null);
    if (default_val) |val| {
        defer val.decref();
        try std.testing.expect(try py.PyList(root).from.check(val));
        try std.testing.expectEqual(@as(usize, 0), py.PyList(root).from.unchecked(val).length());
    }

    // Check that the key now exists
    const contains = try dd_dict.contains("missing");
    try std.testing.expect(contains);

    // Set and get a value normally
    const existing_key = try py.PyString.create("existing");
    defer existing_key.obj.decref();
    const val_to_set = try py.PyLong.create(123);
    defer val_to_set.obj.decref();

    try dd_dict.setItem(existing_key, val_to_set);
    const retrieved_val = try dd_dict.getItem(py.PyLong, "existing");
    try std.testing.expect(retrieved_val != null);
    if (retrieved_val) |val| {
        defer val.obj.decref();
        try std.testing.expectEqual(@as(i64, 123), try val.as(i64));
    }

    // Check default_factory attribute
    const factory = try dd.defaultFactory();
    defer factory.decref();
    try std.testing.expect(factory.py == list_type.py);
}