const std = @import("std");
const py = @import("../pyz3.zig");
const PyObjectMixin = @import("./obj.zig").PyObjectMixin;
const ffi = py.ffi;
const PyError = @import("../errors.zig").PyError;
const State = @import("../discovery.zig").State;

/// Wrapper for Python PyBool.
///
/// See: https://docs.python.org/3/c-api/bool.html
///
/// Note: refcounting semantics apply, even for bools!
pub const PyBool = extern struct {
    obj: py.PyObject,

    const Self = @This();
    pub const from = PyObjectMixin("bool", "PyBool", Self);

    pub fn create(value: bool) !Self {
        return if (value) true_() else false_();
    }

    pub fn asbool(self: Self) bool {
        return ffi.Py_IsTrue(self.obj.py) == 1;
    }

    pub fn intobool(self: Self) bool {
        self.decref();
        return self.asbool();
    }

    pub fn true_() Self {
        return .{ .obj = .{ .py = ffi.PyBool_FromLong(1) } };
    }

    pub fn false_() Self {
        return .{ .obj = .{ .py = ffi.PyBool_FromLong(0) } };
    }
};

test "PyBool" {
    py.initialize();
    defer py.finalize();

    const pytrue = PyBool.true_();
    defer pytrue.obj.decref();

    const pyfalse = PyBool.false_();
    defer pyfalse.obj.decref();

    try std.testing.expect(pytrue.asbool());
    try std.testing.expect(!pyfalse.asbool());
}
