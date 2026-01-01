// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License.
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const std = @import("std");
const ffi = @import("ffi");
const PyObjectMixin = @import("./obj.zig").PyObjectMixin;
const PyError = @import("../errors.zig").PyError;
const builtins = @import("../builtins.zig");
const conversions = @import("../conversions.zig");

// Forward declarations to avoid circular imports
const PyObject = @import("./obj.zig").PyObject;
const PyTuple = @import("./tuple.zig").PyTuple;
const PyDict = @import("./dict.zig").PyDict;
const PyString = @import("./str.zig").PyString;
const PyAwaitable = @import("./awaitable.zig").PyAwaitable;

/// Wrapper for Python async generator
pub fn PyAsyncGenerator(comptime root: type) type {
    return extern struct {
        obj: PyObject,

        const Self = @This();
        pub const from = struct {
            pub fn check(obj: PyObject) !bool {
                const types = try builtins.import(root, "types");
                defer types.decref();
                const async_gen_type = try types.get("AsyncGeneratorType");
                defer async_gen_type.decref();
                return builtins.isinstance(root, obj, async_gen_type);
            }

            pub fn checked(obj: PyObject) !Self {
                if (try from.check(obj) == false) {
                    const typeName = try builtins.str(root, builtins.type_(root, obj));
                    defer typeName.obj.decref();
                    const TypeError = @import("./error.zig").TypeError;
                    return TypeError(root).raiseFmt("expected {s}, found {s}", .{ "async_generator", try typeName.asSlice() });
                }
                return .{ .obj = obj };
            }

            pub fn unchecked(obj: PyObject) Self {
                return .{ .obj = obj };
            }
        };

        /// Check if an object is an async generator.
        pub fn check(obj: PyObject) !bool {
            return from.check(obj);
        }

        /// Returns an awaitable that results in the next value from the generator.
        pub fn anext(self: Self) !PyAwaitable(root) {
            const anext_method = try self.obj.get("__anext__");
            defer anext_method.decref();
            const awaitable = try builtins.call0(root, PyObject, anext_method);
            return PyAwaitable(root){ .obj = awaitable };
        }

        /// Sends a value into the async generator. Returns an awaitable.
        pub fn asend(self: Self, value: PyObject) !PyAwaitable(root) {
            const asend_method = try self.obj.get("asend");
            defer asend_method.decref();
            const awaitable = try builtins.call(root, PyObject, asend_method, .{value}, .{});
            return PyAwaitable(root){ .obj = awaitable };
        }

        /// Throws an exception into the async generator. Returns an awaitable.
        pub fn athrow(self: Self, exc_type: PyObject, value: ?PyObject, traceback: ?PyObject) !PyAwaitable(root) {
            const athrow_method = try self.obj.get("athrow");
            defer athrow_method.decref();

            var args_tuple = try PyTuple(root).new(3);
            defer args_tuple.obj.decref();

            try args_tuple.setOwnedItem(0, exc_type);
            exc_type.incref();

            if (value) |v| {
                try args_tuple.setOwnedItem(1, v);
                v.incref();
            } else {
                try args_tuple.setOwnedItem(1, builtins.None());
            }

            if (traceback) |tb| {
                try args_tuple.setOwnedItem(2, tb);
                tb.incref();
            } else {
                try args_tuple.setOwnedItem(2, builtins.None());
            }

            const awaitable = try builtins.call(root, PyObject, athrow_method, args_tuple.obj, .{});
            return PyAwaitable(root){ .obj = awaitable };
        }
    };
}

test "PyAsyncGenerator" {
    const testing_module = @import("../testing.zig");
    var fixture = testing_module.TestFixture.init();
    defer fixture.deinit();

    fixture.initPython();
    const root = @This();

    const code = "async def my_agen():\n    yield 1\n    yield 2\n\nagen = my_agen()";

    const builtins_mod = try builtins.import(root, "builtins");
    defer builtins_mod.decref();
    const exec = try builtins_mod.get("exec");
    defer exec.decref();
    const globals = try PyDict(root).new();
    defer globals.obj.decref();

    _ = try builtins.call(root, PyObject, exec, .{ try PyString(root).create(code), globals.obj }, .{});

    const agen_obj = try globals.getItem(PyObject, "agen") orelse unreachable;
    defer agen_obj.decref();

    const agen = PyAsyncGenerator(root).from.unchecked(agen_obj);

    // anext 1
    var awaitable1 = try agen.anext();
    var result1 = awaitable1.await_() catch |err| {
        // Handle StopAsyncIteration
        var ptype: ?*ffi.PyObject = undefined;
        var pvalue: ?*ffi.PyObject = undefined;
        var ptraceback: ?*ffi.PyObject = undefined;
        ffi.PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        const stop_iteration_type = ffi.PyExc_StopAsyncIteration;
        if (ptype != null and ffi.PyErr_GivenExceptionMatches(ptype, stop_iteration_type) != 0) {
            // Error already cleared by PyErr_Fetch
            unreachable; // Should not stop here
        }
        ffi.PyErr_Restore(ptype, pvalue, ptraceback);
        return err;
    };
    defer result1.decref();
    try std.testing.expectEqual(@as(i64, 1), try conversions.as(root, i64, result1));

    // anext 2
    var awaitable2 = try agen.anext();
    var result2 = awaitable2.await_() catch |err| {
        var ptype: ?*ffi.PyObject = undefined;
        var pvalue: ?*ffi.PyObject = undefined;
        var ptraceback: ?*ffi.PyObject = undefined;
        ffi.PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        const stop_iteration_type = ffi.PyExc_StopAsyncIteration;
        if (ptype != null and ffi.PyErr_GivenExceptionMatches(ptype, stop_iteration_type) != 0) {
            // Error already cleared by PyErr_Fetch
            unreachable; // Should not stop here
        }
        ffi.PyErr_Restore(ptype, pvalue, ptraceback);
        return err;
    };
    defer result2.decref();
    try std.testing.expectEqual(@as(i64, 2), try conversions.as(root, i64, result2));

    // anext 3 - expecting StopAsyncIteration
    var awaitable3 = try agen.anext();
    awaitable3.await_() catch |err| {
        var ptype: ?*ffi.PyObject = undefined;
        var pvalue: ?*ffi.PyObject = undefined;
        var ptraceback: ?*ffi.PyObject = undefined;
        ffi.PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        const stop_iteration_type = ffi.PyExc_StopAsyncIteration;
        if (ptype != null and ffi.PyErr_GivenExceptionMatches(ptype, stop_iteration_type) != 0) {
            // Error already cleared by PyErr_Fetch
        } else {
            ffi.PyErr_Restore(ptype, pvalue, ptraceback);
            return err;
        }
    };
}