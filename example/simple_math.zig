// Simple example for zigimport demonstration
const py = @import("pyz3");

pub fn add(args: struct { a: i64, b: i64 }) i64 {
    return args.a + args.b;
}

pub fn multiply(args: struct { a: i64, b: i64 }) i64 {
    return args.a * args.b;
}

pub fn fibonacci(args: struct { n: u64 }) u64 {
    if (args.n < 2) return args.n;

    var sum: u64 = 0;
    var last: u64 = 0;
    var curr: u64 = 1;
    var i: u64 = 1;
    while (i < args.n) : (i += 1) {
        sum = last + curr;
        last = curr;
        curr = sum;
    }
    return sum;
}

comptime {
    py.rootmodule(@This());
}
