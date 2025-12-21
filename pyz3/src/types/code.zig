const std = @import("std");
const py = @import("../pyz3.zig");
const State = @import("../discovery.zig").State;

const ffi = py.ffi;

/// Wrapper for Python PyCode.
/// See: https://docs.python.org/3/c-api/code.html
pub fn PyCode(comptime root: type) type {
    return extern struct {
        obj: py.PyObject,

        const Self = @This();

        pub inline fn firstLineNumber(self: *const Self) !u32 {
            const lineNo = try py.as(root, py.PyLong, try self.obj.get("co_firstlineno"));
            defer lineNo.obj.decref();
            return lineNo.as(u32);
        }

        pub inline fn fileName(self: *const Self) !py.PyString {
            return try py.as(root, py.PyString, try self.obj.get("co_filename"));
        }

        pub inline fn name(self: *const Self) !py.PyString {
            return try py.as(root, py.PyString, try self.obj.get("co_name"));
        }
    };
}

test "PyCode" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    const pf = py.PyFrame(root).get();
    try std.testing.expectEqual(@as(?py.PyFrame(root), null), pf);
}
