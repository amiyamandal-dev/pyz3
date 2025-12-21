// Utility functions - will be tracked as dependency
const helpers = @import("helpers.zig");

pub fn transform(x: i64) i64 {
    return helpers.double(x) + 10;
}

pub fn multiply_by_two(x: i64) i64 {
    return helpers.double(x);
}
