const std = @import("std");
const py = @import("../pyz3.zig");
const PyObjectMixin = @import("./obj.zig").PyObjectMixin;
const ffi = py.ffi;
const PyError = @import("../errors.zig").PyError;
const State = @import("../discovery.zig").State;

pub const PyBytes = extern struct {
    obj: py.PyObject,

    const Self = @This();
    pub const from = PyObjectMixin("bytes", "PyBytes", Self);

    pub fn create(value: []const u8) !Self {
        const bytes = ffi.PyBytes_FromStringAndSize(value.ptr, @intCast(value.len)) orelse return PyError.PyRaised;
        return .{ .obj = .{ .py = bytes } };
    }

    /// Return the bytes representation of object obj that implements the buffer protocol.
    pub fn fromObject(obj: anytype) !Self {
        const pyobj = py.object(obj);
        const bytes = ffi.PyBytes_FromObject(pyobj.py) orelse return PyError.PyRaised;
        return .{ .obj = .{ .py = bytes } };
    }

    /// Return the length of the bytes object.
    pub fn length(self: Self) !usize {
        return @intCast(ffi.PyBytes_Size(self.obj.py));
    }

    /// Returns a view over the PyBytes bytes.
    pub fn asSlice(self: Self) ![:0]const u8 {
        var buffer: [*]u8 = undefined;
        var size: i64 = 0;
        if (ffi.PyBytes_AsStringAndSize(self.obj.py, @ptrCast(&buffer), &size) < 0) {
            return PyError.PyRaised;
        }
        return buffer[0..@as(usize, @intCast(size)) :0];
    }
};

const testing = std.testing;

test "PyBytes" {
    py.initialize();
    defer py.finalize();

    const a = "Hello";

    var ps = try PyBytes.create(a);
    defer ps.obj.decref();

    const ps_slice = try ps.asSlice();
    try testing.expectEqual(a.len, ps_slice.len);
    try testing.expectEqual(a.len, try ps.length());
    try testing.expectEqual(@as(u8, 0), ps_slice[ps_slice.len]);

    try testing.expectEqualStrings("Hello", ps_slice);
}
