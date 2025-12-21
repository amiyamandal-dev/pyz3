// Example showing dependency tracking
const py = @import("pyz3");
const utils = @import("utils.zig");
const math = @import("math.zig");

pub fn process(args: struct { x: i64 }) i64 {
    const transformed = utils.transform(args.x);
    return math.calculate(transformed);
}

pub fn combined(args: struct { a: i64, b: i64 }) i64 {
    return math.add(args.a, args.b) * utils.multiply_by_two(1);
}

comptime {
    py.rootmodule(@This());
}
