const std = @import("std");
const py = @import("../pyz3.zig");
const State = @import("../discovery.zig").State;

const ffi = py.ffi;

/// Wrapper for Python PyFrame.
/// See: https://docs.python.org/3/c-api/frame.html
pub fn PyFrame(comptime root: type) type {
    return extern struct {
        obj: py.PyObject,
        const Self = @This();

        pub fn get() ?Self {
            const frame = ffi.PyEval_GetFrame();
            return if (frame) |f| .{ .obj = .{ .py = objPtr(f) } } else null;
        }

        pub fn code(self: Self) py.PyCode(root) {
            const codeObj = ffi.PyFrame_GetCode(framePtr(self.obj.py));
            return .{ .obj = .{ .py = @ptrCast(@alignCast(codeObj)) } };
        }

        pub inline fn lineNumber(self: Self) u32 {
            return @intCast(ffi.PyFrame_GetLineNumber(framePtr(self.obj.py)));
        }

        inline fn framePtr(obj: *ffi.PyObject) *ffi.PyFrameObject {
            return @ptrCast(@alignCast(obj));
        }

        inline fn objPtr(obj: *ffi.PyFrameObject) *ffi.PyObject {
            return @ptrCast(@alignCast(obj));
        }
    };
}

test "PyFrame" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    const pf = PyFrame(root).get();
    try std.testing.expectEqual(@as(?PyFrame(root), null), pf);
}
