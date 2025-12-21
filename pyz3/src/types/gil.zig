const std = @import("std");
const py = @import("../pyz3.zig");

const ffi = py.ffi;
const PyError = @import("../errors.zig").PyError;

pub const PyGIL = extern struct {
    state: ffi.PyGILState_STATE,

    /// Acqiure the GIL. Ensure to call `release` when done, e.g. using `defer gil.release()`.
    pub fn ensure() PyGIL {
        return .{ .state = ffi.PyGILState_Ensure() };
    }

    pub fn release(self: PyGIL) void {
        ffi.PyGILState_Release(self.state);
    }
};
