const std = @import("std");
const py = @import("pyz3");

const root = @This();

// --8<-- [start:example]
test "pyz3 pytest" {
    py.initialize();
    defer py.finalize();

    const str = try py.PyString.create("hello");
    defer str.obj.decref();

    try std.testing.expectEqualStrings("hello", try str.asSlice());
}
// --8<-- [end:example]

test "pyz3-expected-failure" {
    py.initialize();
    defer py.finalize();

    const str = try py.PyString.create("hello");
    defer str.obj.decref();

    try std.testing.expectEqualStrings("world", try str.asSlice());
}

comptime {
    py.rootmodule(root);
}
