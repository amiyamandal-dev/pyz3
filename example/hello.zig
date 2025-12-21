// --8<-- [start:ex]
const py = @import("pyz3");

const root = @This();

pub fn hello() !py.PyString {
    return try py.PyString.create("Hello!");
}

comptime {
    py.rootmodule(root);
}
// --8<-- [end:ex]
