const std = @import("std");
const py = @import("../pyz3.zig");
const PyObjectMixin = @import("./obj.zig").PyObjectMixin;
const ffi = py.ffi;
const PyError = @import("../errors.zig").PyError;
const State = @import("../discovery.zig").State;

/// Wrapper for Python PySlice.
pub fn PySlice(comptime root: type) type {
    return extern struct {
        obj: py.PyObject,

        const Self = @This();
        pub const from = PyObjectMixin("slice", "PySlice", Self);

        pub fn create(start: anytype, stop: anytype, step: anytype) !Self {
            // Note: This manual optional handling is verbose but correct. A helper function could
            // simplify this pattern: optionalCreate(root, value) -> ?*PyObject
            // However, the current approach is clear and avoids hidden allocations/decrefs.
            const pystart = if (@typeInfo(@TypeOf(start)) == .null) null else (try py.create(root, start)).py;
            defer if (@typeInfo(@TypeOf(start)) != .null) py.decref(root, pystart);
            const pystop = if (@typeInfo(@TypeOf(stop)) == .null) null else (try py.create(root, stop)).py;
            defer if (@typeInfo(@TypeOf(stop)) != .null) py.decref(root, pystop);
            const pystep = if (@typeInfo(@TypeOf(step)) == .null) null else (try py.create(root, step)).py;
            defer if (@typeInfo(@TypeOf(step)) != .null) py.decref(root, pystep);

            const pyslice = ffi.PySlice_New(pystart, pystop, pystep) orelse return PyError.PyRaised;
            return .{ .obj = .{ .py = pyslice } };
        }

        pub fn getStart(self: Self, comptime T: type) !T {
            return try py.as(root, T, try self.obj.get("start"));
        }

        pub fn getStop(self: Self, comptime T: type) !T {
            return try py.as(root, T, try self.obj.get("stop"));
        }

        pub fn getStep(self: Self, comptime T: type) !T {
            return try py.as(root, T, try self.obj.get("step"));
        }
    };
}

test "PySlice" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    const range = try PySlice(root).create(0, 100, null);
    defer range.obj.decref();

    try std.testing.expectEqual(@as(u64, 0), try range.getStart(u64));
    try std.testing.expectEqual(@as(u64, 100), try range.getStop(u64));
    try std.testing.expectEqual(@as(?u64, null), try range.getStep(?u64));
}
