const py = @import("pyz3");
const c = @cImport({
    @cInclude("myheader.h");
});

pub fn custom_function(args: struct { x: i64 }) i64 {
    // Calls C function from native.c
    return c.native_multiply(args.x, 2);
}

comptime {
    py.rootmodule(@This());
}
