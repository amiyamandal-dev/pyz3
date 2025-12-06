// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const std = @import("std");
const ffi = @import("ffi");
const py = @import("../pydust.zig");
const PyError = @import("../errors.zig").PyError;

/// Python Decimal object wrapper for precise decimal arithmetic
pub const PyDecimal = extern struct {
    obj: py.PyObject,

    const Self = @This();

    /// Create a Decimal from an integer
    pub fn fromInt(value: i64) !Self {
        const decimal_mod = try py.import("decimal");
        defer decimal_mod.decref();

        const decimal_class = try decimal_mod.getAttribute("Decimal");
        defer decimal_class.decref();

        const val = try py.PyLong.from(value);
        defer val.obj.decref();

        const decimal_obj = try py.call(@import("../pydust.zig"), decimal_class, .{val.obj});
        return .{ .obj = decimal_obj };
    }

    /// Create a Decimal from a string
    pub fn fromString(value: []const u8) !Self {
        const decimal_mod = try py.import("decimal");
        defer decimal_mod.decref();

        const decimal_class = try decimal_mod.getAttribute("Decimal");
        defer decimal_class.decref();

        const val = try py.PyString.create(value);
        defer val.obj.decref();

        const decimal_obj = try py.call(@import("../pydust.zig"), decimal_class, .{val.obj});
        return .{ .obj = decimal_obj };
    }

    /// Create a Decimal from a float (note: may have precision issues)
    pub fn fromFloat(value: f64) !Self {
        const decimal_mod = try py.import("decimal");
        defer decimal_mod.decref();

        const decimal_class = try decimal_mod.getAttribute("Decimal");
        defer decimal_class.decref();

        const val = try py.PyFloat.from(value);
        defer val.obj.decref();

        const decimal_obj = try py.call(@import("../pydust.zig"), decimal_class, .{val.obj});
        return .{ .obj = decimal_obj };
    }

    /// Check if object is a Decimal
    pub fn check(obj: py.PyObject) bool {
        const decimal_mod = py.import("decimal") catch return false;
        defer decimal_mod.decref();

        const decimal_class = decimal_mod.getAttribute("Decimal") catch return false;
        defer decimal_class.decref();

        return py.isinstance(@import("../pydust.zig"), obj, decimal_class) catch false;
    }

    /// Convert to float
    pub fn toFloat(self: Self) !f64 {
        const builtins = try py.import("builtins");
        defer builtins.decref();

        const float_fn = try builtins.getAttribute("float");
        defer float_fn.decref();

        const result = try py.call(@import("../pydust.zig"), float_fn, .{self.obj});
        defer result.decref();

        return try py.as(f64, @import("../pydust.zig"), result);
    }

    /// Convert to integer
    pub fn toInt(self: Self) !i64 {
        const builtins = try py.import("builtins");
        defer builtins.decref();

        const int_fn = try builtins.getAttribute("int");
        defer int_fn.decref();

        const result = try py.call(@import("../pydust.zig"), int_fn, .{self.obj});
        defer result.decref();

        return try py.as(i64, @import("../pydust.zig"), result);
    }

    /// Convert to string
    pub fn toString(self: Self) !py.PyString {
        const str_fn = try py.import("builtins").getAttribute("str");
        defer str_fn.decref();

        const result = try py.call(@import("../pydust.zig"), str_fn, .{self.obj});
        return .{ .obj = result };
    }

    /// Add two Decimals
    pub fn add(self: Self, other: Self) !Self {
        const add_method = try self.obj.getAttribute("__add__");
        defer add_method.decref();

        const result = try py.call(@import("../pydust.zig"), add_method, .{other.obj});
        return .{ .obj = result };
    }

    /// Subtract two Decimals
    pub fn sub(self: Self, other: Self) !Self {
        const sub_method = try self.obj.getAttribute("__sub__");
        defer sub_method.decref();

        const result = try py.call(@import("../pydust.zig"), sub_method, .{other.obj});
        return .{ .obj = result };
    }

    /// Multiply two Decimals
    pub fn mul(self: Self, other: Self) !Self {
        const mul_method = try self.obj.getAttribute("__mul__");
        defer mul_method.decref();

        const result = try py.call(@import("../pydust.zig"), mul_method, .{other.obj});
        return .{ .obj = result };
    }

    /// Divide two Decimals
    pub fn div(self: Self, other: Self) !Self {
        const div_method = try self.obj.getAttribute("__truediv__");
        defer div_method.decref();

        const result = try py.call(@import("../pydust.zig"), div_method, .{other.obj});
        return .{ .obj = result };
    }

    /// Floor division
    pub fn floorDiv(self: Self, other: Self) !Self {
        const div_method = try self.obj.getAttribute("__floordiv__");
        defer div_method.decref();

        const result = try py.call(@import("../pydust.zig"), div_method, .{other.obj});
        return .{ .obj = result };
    }

    /// Modulo
    pub fn mod(self: Self, other: Self) !Self {
        const mod_method = try self.obj.getAttribute("__mod__");
        defer mod_method.decref();

        const result = try py.call(@import("../pydust.zig"), mod_method, .{other.obj});
        return .{ .obj = result };
    }

    /// Power (self ** other)
    pub fn pow(self: Self, other: Self) !Self {
        const pow_method = try self.obj.getAttribute("__pow__");
        defer pow_method.decref();

        const result = try py.call(@import("../pydust.zig"), pow_method, .{other.obj});
        return .{ .obj = result };
    }

    /// Absolute value
    pub fn abs(self: Self) !Self {
        const abs_method = try self.obj.getAttribute("__abs__");
        defer abs_method.decref();

        const result = try py.call0(@import("../pydust.zig"), abs_method);
        return .{ .obj = result };
    }

    /// Negation
    pub fn neg(self: Self) !Self {
        const neg_method = try self.obj.getAttribute("__neg__");
        defer neg_method.decref();

        const result = try py.call0(@import("../pydust.zig"), neg_method);
        return .{ .obj = result };
    }

    /// Compare for equality
    pub fn eq(self: Self, other: Self) !bool {
        const eq_method = try self.obj.getAttribute("__eq__");
        defer eq_method.decref();

        const result = try py.call(@import("../pydust.zig"), eq_method, .{other.obj});
        defer result.decref();

        return try py.as(bool, @import("../pydust.zig"), result);
    }

    /// Compare less than
    pub fn lt(self: Self, other: Self) !bool {
        const lt_method = try self.obj.getAttribute("__lt__");
        defer lt_method.decref();

        const result = try py.call(@import("../pydust.zig"), lt_method, .{other.obj});
        defer result.decref();

        return try py.as(bool, @import("../pydust.zig"), result);
    }

    /// Compare less than or equal
    pub fn le(self: Self, other: Self) !bool {
        const le_method = try self.obj.getAttribute("__le__");
        defer le_method.decref();

        const result = try py.call(@import("../pydust.zig"), le_method, .{other.obj});
        defer result.decref();

        return try py.as(bool, @import("../pydust.zig"), result);
    }

    /// Round to n decimal places
    pub fn round(self: Self, places: i64) !Self {
        const quantize_method = try self.obj.getAttribute("quantize");
        defer quantize_method.decref();

        // Create quantization value: Decimal(10) ** -places
        const ten = try Self.fromInt(10);
        defer ten.obj.decref();

        const neg_places = try Self.fromInt(-places);
        defer neg_places.obj.decref();

        const quantum = try ten.pow(neg_places);
        defer quantum.obj.decref();

        const result = try py.call(@import("../pydust.zig"), quantize_method, .{quantum.obj});
        return .{ .obj = result };
    }

    /// Square root
    pub fn sqrt(self: Self) !Self {
        const sqrt_method = try self.obj.getAttribute("sqrt");
        defer sqrt_method.decref();

        const result = try py.call0(@import("../pydust.zig"), sqrt_method);
        return .{ .obj = result };
    }

    /// Natural logarithm
    pub fn ln(self: Self) !Self {
        const ln_method = try self.obj.getAttribute("ln");
        defer ln_method.decref();

        const result = try py.call0(@import("../pydust.zig"), ln_method);
        return .{ .obj = result };
    }

    /// Base-10 logarithm
    pub fn log10(self: Self) !Self {
        const log10_method = try self.obj.getAttribute("log10");
        defer log10_method.decref();

        const result = try py.call0(@import("../pydust.zig"), log10_method);
        return .{ .obj = result };
    }

    /// Exponential function (e^x)
    pub fn exp(self: Self) !Self {
        const exp_method = try self.obj.getAttribute("exp");
        defer exp_method.decref();

        const result = try py.call0(@import("../pydust.zig"), exp_method);
        return .{ .obj = result };
    }

    /// Check if the decimal is finite
    pub fn isFinite(self: Self) !bool {
        const method = try self.obj.getAttribute("is_finite");
        defer method.decref();

        const result = try py.call0(@import("../pydust.zig"), method);
        defer result.decref();

        return try py.as(bool, @import("../pydust.zig"), result);
    }

    /// Check if the decimal is infinite
    pub fn isInfinite(self: Self) !bool {
        const method = try self.obj.getAttribute("is_infinite");
        defer method.decref();

        const result = try py.call0(@import("../pydust.zig"), method);
        defer result.decref();

        return try py.as(bool, @import("../pydust.zig"), result);
    }

    /// Check if the decimal is NaN
    pub fn isNaN(self: Self) !bool {
        const method = try self.obj.getAttribute("is_nan");
        defer method.decref();

        const result = try py.call0(@import("../pydust.zig"), method);
        defer result.decref();

        return try py.as(bool, @import("../pydust.zig"), result);
    }
};
