// --8<-- [start:ex]
pub const __doc__ =
    \\Zig multi-line strings make it easy to define a docstring...
    \\
    \\..with lots of lines!
    \\
    \\P.S. I'm sure one day we'll hook into Zig's AST and read the Zig doc comments ;)
;

const std = @import("std");
const py = @import("pyz3");

const root = @This();
const Self = root; // (1)!

count_: u32 = 0, // (2)!
name: py.PyString,

pub fn __init__(self: *Self) !void { // (3)!
    self.* = .{ .name = try py.PyString.create("Ziggy") };
}

pub fn __del__(self: Self) void {
    self.name.obj.decref();
}

pub fn increment(self: *Self) void { // (4)!
    self.count_ += 1;
}

pub fn count(self: *const Self) u32 {
    return self.count_;
}

pub fn whoami(self: *const Self) py.PyString {
    py.incref(root, self.name);
    return self.name;
}

pub fn hello(
    self: *const Self,
    args: struct { name: py.PyString }, // (5)!
) !py.PyString {
    return py.PyString.createFmt(
        "Hello, {s}. It's {s}",
        .{ try args.name.asSlice(), try self.name.asSlice() },
    );
}

pub const submod = py.module(struct { // (6)!
    pub fn world() !py.PyString {
        return try py.PyString.create("Hello, World!");
    }
});

comptime {
    py.rootmodule(root);
} // (7)!
// --8<-- [end:ex]
