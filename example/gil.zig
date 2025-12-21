const std = @import("std");
const py = @import("pyz3");

const root = @This();

// --8<-- [start:gil]
pub fn sleep(args: struct { millis: u64 }) void {
    std.Thread.sleep(args.millis * 1_000_000);
}

pub fn sleep_release(args: struct { millis: u64 }) void {
    const nogil = py.nogil();
    defer nogil.acquire();
    std.Thread.sleep(args.millis * 1_000_000);
}
// --8<-- [end:gil]

comptime {
    py.rootmodule(root);
}
