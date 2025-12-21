const std = @import("std");
const py = @import("pyz3");

const root = @This();

// --8<-- [start:valueerror]
pub fn raise_value_error(args: struct { message: py.PyString }) !void {
    return py.ValueError(root).raise(try args.message.asSlice());
}
// --8<-- [end:valueerror]

pub const CustomError = error{Oops};

pub fn raise_custom_error() !void {
    return CustomError.Oops;
}

comptime {
    py.rootmodule(root);
}
