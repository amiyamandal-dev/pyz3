const std = @import("std");
const py = @import("../pyz3.zig");
const PyObjectMixin = @import("./obj.zig").PyObjectMixin;
const ffi = py.ffi;
const PyError = @import("../errors.zig").PyError;
const State = @import("../discovery.zig").State;

/// Wrapper for Python PyIter.
/// Constructed using py.iter(...)
pub fn PyIter(comptime root: type) type {
    return extern struct {
        obj: py.PyObject,

        const Self = @This();
        pub const from = PyObjectMixin("iterator", "PyIter", Self);

        pub fn next(self: Self, comptime T: type) !?T {
            if (ffi.PyIter_Next(self.obj.py)) |result| {
                return try py.as(root, T, result);
            }

            // If no exception, then the item is missing.
            if (ffi.PyErr_Occurred() == null) {
                return null;
            }

            return PyError.PyRaised;
        }

        // Note: PyIter_Send is used for async generators and coroutines (PEP 525).
        // Implementation would add: pub fn send(self: Self, value: anytype) !py.PyObject
        // wrapping ffi.PyIter_Send(). Add when async generator support is needed.
    };
}

test "PyIter" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    const tuple = try py.PyTuple(root).create(.{ 1, 2, 3 });
    defer tuple.obj.decref();

    const iterator = try py.iter(root, tuple);
    var previous: u64 = 0;
    while (try iterator.next(u64)) |v| {
        try std.testing.expect(v > previous);
        previous = v;
    }
}
