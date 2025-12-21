const std = @import("std");
const py = @import("pyz3");

const root = @This();

pub const Range = py.class(struct {
    pub const __doc__ = "An example of iterable class";

    const Self = @This();

    lower: i64,
    upper: i64,
    step: i64,

    pub fn __init__(self: *Self, args: struct { lower: i64, upper: i64, step: i64 }) void {
        self.* = .{ .lower = args.lower, .upper = args.upper, .step = args.step };
    }

    pub fn __iter__(self: *const Self) !*RangeIterator.definition {
        return try py.init(root, RangeIterator.definition, .{ .next = self.lower, .stop = self.upper, .step = self.step });
    }
});

pub const RangeIterator = py.class(struct {
    pub const __doc__ = "Range iterator";

    const Self = @This();

    next: i64,
    stop: i64,
    step: i64,

    pub fn __next__(self: *Self) ?i64 {
        if (self.next >= self.stop) {
            return null;
        }
        defer self.next += self.step;
        return self.next;
    }
});

comptime {
    py.rootmodule(root);
}
