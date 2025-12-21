const std = @import("std");
const py = @import("../pyz3.zig");
const PyObjectMixin = @import("./obj.zig").PyObjectMixin;
const ffi = py.ffi;
const PyError = @import("../errors.zig").PyError;
const State = @import("../discovery.zig").State;

/// Wrapper for Python PyFloat.
/// See: https://docs.python.org/3/c-api/float.html
pub const PyFloat = extern struct {
    obj: py.PyObject,

    const Self = @This();
    pub const from = PyObjectMixin("float", "PyFloat", Self);

    pub fn create(value: anytype) !Self {
        const pyfloat = ffi.PyFloat_FromDouble(@floatCast(value)) orelse return PyError.PyRaised;
        return .{ .obj = .{ .py = pyfloat } };
    }

    pub fn as(self: Self, comptime T: type) !T {
        return switch (T) {
            f16, f32, f64 => {
                const double = ffi.PyFloat_AsDouble(self.obj.py);
                if (ffi.PyErr_Occurred() != null) return PyError.PyRaised;
                return @floatCast(double);
            },
            else => @compileError("Unsupported float type " ++ @typeName(T)),
        };
    }
};

test "PyFloat" {
    py.initialize();
    defer py.finalize();

    const pf = try PyFloat.create(1.0);
    defer pf.obj.decref();

    try std.testing.expectEqual(@as(f32, 1.0), try pf.as(f32));
}
