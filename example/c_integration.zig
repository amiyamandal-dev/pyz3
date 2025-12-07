// Example demonstrating C/C++ integration with pyz3
// This module wraps C functions and exposes them to Python

const py = @import("pyz3");
const std = @import("std");

// Import the C helper functions
const c = @cImport({
    @cInclude("c_math_helper.h");
});

/// Add two integers using C implementation
pub fn add(args: struct { a: i32, b: i32 }) i32 {
    return c.c_add(@intCast(args.a), @intCast(args.b));
}

/// Multiply two integers using C implementation
pub fn multiply(args: struct { a: i32, b: i32 }) i32 {
    return c.c_multiply(@intCast(args.a), @intCast(args.b));
}

/// Divide two floats using C implementation
pub fn divide(args: struct { a: f64, b: f64 }) !f64 {
    const result = c.c_divide(args.a, args.b);
    if (result == 0.0 and args.b == 0.0) {
        return error.ZeroDivisionError;
    }
    return result;
}

/// Calculate factorial using C implementation
pub fn factorial(args: struct { n: i32 }) !i32 {
    if (args.n < 0) {
        return error.ValueError;
    }
    if (args.n > 20) {
        return error.ValueError;
    }
    return c.c_factorial(@intCast(args.n));
}

/// Calculate Fibonacci number using C implementation
pub fn fibonacci(args: struct { n: i32 }) !i32 {
    if (args.n < 0) {
        return error.ValueError;
    }
    if (args.n > 30) {
        return error.ValueError;
    }
    return c.c_fibonacci(@intCast(args.n));
}

comptime {
    py.rootmodule(@This());
}
