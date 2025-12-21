const std = @import("std");
const py = @import("pyz3");

const root = @This();

pub fn pyobject() !py.PyObject {
    return (try py.PyString.create("hello")).obj;
}

pub fn pystring() !py.PyString {
    return py.PyString.create("hello world");
}

pub fn zigvoid() void {}

pub fn zigbool() bool {
    return true;
}

pub fn zigu32() u32 {
    return 32;
}

pub fn zigu64() u64 {
    return 8589934592;
}

// Note: u128/i128 support would require using Python's arbitrary precision integers.
// Python's int can handle any size, but the conversion would need PyLong_FromString()
// or manual digit manipulation since PyLong_FromLongLong() is limited to 64 bits.
// Example implementation would convert the u128 to a decimal string first.
// pub fn zigu128() u128 {
//     return 9223372036854775809;
// }

pub fn zigi32() i32 {
    return -32;
}

pub fn zigi64() i64 {
    return -8589934592;
}

// Note: i128 support would require using Python's arbitrary precision integers.
// Similar to u128, this would need PyLong_FromString() with sign handling.
// pub fn zigi128() i128 {
//     return -9223372036854775809;
// }

pub fn zigf16() f16 {
    return 32720.0;
}

pub fn zigf32() f32 {
    return 2.71 * std.math.pow(f32, 10, 38);
}

pub fn zigf64() f64 {
    return 2.71 * std.math.pow(f64, 10, 39);
}

const TupleResult = struct { py.PyObject, u64 };

pub fn zigtuple() !TupleResult {
    return .{ py.object(root, try py.PyString.create("hello")), 128 };
}

const StructResult = struct { foo: u64, bar: bool };

pub fn zigstruct() StructResult {
    return .{ .foo = 1234, .bar = true };
}

comptime {
    py.rootmodule(root);
}
