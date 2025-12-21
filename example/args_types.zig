const std = @import("std");
const py = @import("pyz3");

const root = @This();

const ArgStruct = struct {
    foo: i32,
    bar: bool,
};

pub fn zigstruct(args: struct { x: ArgStruct }) bool {
    return args.x.foo == 1234 and args.x.bar;
}

comptime {
    py.rootmodule(root);
}
