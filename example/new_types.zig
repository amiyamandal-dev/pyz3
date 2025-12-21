/// Example demonstrating all new Python type wrappers.
///
/// This module showcases:
/// - PySet / PyFrozenSet: Set operations
/// - PyComplex: Complex number arithmetic
/// - PyByteArray: Mutable byte sequences
/// - PyRange: Range objects
/// - PyGenerator: Generator iteration
/// - PyDateTime/PyDate/PyTime/PyTimeDelta: Date and time operations
/// - PyDecimal: Precise decimal arithmetic
/// - PyPath: File system path operations
/// - PyUUID: UUID generation and manipulation
const std = @import("std");
const py = @import("pyz3");

const root = @This();

// ============================================================================
// PySet and PyFrozenSet Examples
// ============================================================================

/// Create and manipulate a set
pub fn set_operations() !py.PyObject {
    // Create a new set
    const my_set = try py.PySet(root).new();
    defer my_set.obj.decref();

    // Add items
    const item1 = try py.create(root, @as(i64, 1));
    defer item1.decref();
    try my_set.add(item1);

    const item2 = try py.create(root, @as(i64, 2));
    defer item2.decref();
    try my_set.add(item2);

    const item3 = try py.create(root, @as(i64, 3));
    defer item3.decref();
    try my_set.add(item3);

    // Check membership
    const contains = try my_set.contains(item2);
    std.debug.print("Set contains 2: {}\n", .{contains});

    // Get size
    const size = try my_set.len();
    std.debug.print("Set size: {}\n", .{size});

    my_set.obj.incref();
    return my_set.obj;
}

/// Demonstrate set operations (union, intersection, difference)
pub fn set_algebra() !py.PyObject {
    const set_a = try py.PySet(root).fromSlice(&[_]i64{ 1, 2, 3, 4 });
    defer set_a.obj.decref();

    const set_b = try py.PySet(root).fromSlice(&[_]i64{ 3, 4, 5, 6 });
    defer set_b.obj.decref();

    // Union: {1, 2, 3, 4, 5, 6}
    const union_set = try set_a.unionWith(set_b);
    defer union_set.obj.decref();

    // Intersection: {3, 4}
    const intersection = try set_a.intersection(set_b);
    defer intersection.obj.decref();

    // Difference: {1, 2}
    const difference = try set_a.difference(set_b);

    return difference.obj;
}

// ============================================================================
// PyComplex Examples
// ============================================================================

/// Demonstrate complex number arithmetic
pub fn complex_math() !py.PyObject {
    // Create complex numbers
    const a = try py.PyComplex.create(3.0, 4.0); // 3 + 4j
    defer a.obj.decref();

    const b = try py.PyComplex.create(1.0, 2.0); // 1 + 2j
    defer b.obj.decref();

    // Addition
    const sum = try a.add(b); // (4 + 6j)
    defer sum.obj.decref();

    // Multiplication
    const product = try a.mul(b);
    defer product.obj.decref();

    // Absolute value
    const magnitude = try a.abs();
    std.debug.print("Magnitude of 3+4j: {d:.2}\n", .{magnitude});

    // Conjugate
    const conj = try a.conjugate(); // 3 - 4j
    defer conj.obj.decref();

    // Polar coordinates
    const polar = try a.toPolar();
    std.debug.print("Polar: r={d:.2}, phi={d:.2}\n", .{ polar.r, polar.phi });

    conj.obj.incref();
    return conj.obj;
}

// ============================================================================
// PyByteArray Examples
// ============================================================================

/// Demonstrate mutable byte array operations
pub fn bytearray_operations() !py.PyObject {
    // Create from bytes
    const ba = try py.PyByteArray.from("Hello");
    defer ba.obj.decref();

    // Append bytes
    try ba.extend(" World");

    // Get as slice and modify
    const slice = try ba.asSlice();
    slice[0] = 'h'; // Change 'H' to 'h'

    // Reverse in place
    try ba.reverse();

    ba.obj.incref();
    return ba.obj;
}

// ============================================================================
// PyRange Examples
// ============================================================================

/// Demonstrate range operations
pub fn range_operations() !py.PyObject {
    // Create range(10)
    const r = try py.PyRange.new(10);
    defer r.obj.decref();

    const length = try r.len();
    std.debug.print("Range length: {}\n", .{length});

    // Create range(5, 15, 2)
    const r2 = try py.PyRange.fromStartStopStep(5, 15, 2);
    defer r2.obj.decref();

    // Check membership
    const has_7 = try r2.contains(7);
    const has_8 = try r2.contains(8);
    std.debug.print("Range contains 7: {}, contains 8: {}\n", .{ has_7, has_8 });

    r2.obj.incref();
    return r2.obj;
}

// ============================================================================
// PyDateTime Examples
// ============================================================================

/// Demonstrate datetime operations
pub fn datetime_operations() !py.PyObject {
    // Get current datetime
    const now = try py.PyDateTime.now();
    defer now.obj.decref();

    const components = try now.components();
    std.debug.print("Year: {}, Month: {}, Day: {}\n", .{ components.year, components.month, components.day });

    // Create specific datetime
    const dt = try py.PyDateTime.create(2025, 12, 4, 10, 30, 0, 0);
    defer dt.obj.decref();

    // Format as ISO string
    const iso = try dt.isoformat();
    defer iso.obj.decref();

    const iso_str = try py.PyString.asSlice(iso.obj);
    defer iso_str.decref();
    std.debug.print("ISO format: {s}\n", .{iso_str.buf});

    // Create date only
    const today = try py.PyDate.today();
    defer today.obj.decref();

    // Create time delta
    const delta = try py.PyTimeDelta.create(1, 3600, 0); // 1 day + 1 hour
    defer delta.obj.decref();

    const total_secs = try delta.totalSeconds();
    std.debug.print("Total seconds: {d}\n", .{total_secs});

    today.obj.incref();
    return today.obj;
}

// ============================================================================
// PyDecimal Examples
// ============================================================================

/// Demonstrate precise decimal arithmetic
pub fn decimal_precision() !py.PyObject {
    // Create decimals
    const a = try py.PyDecimal.fromString("0.1");
    defer a.obj.decref();

    const b = try py.PyDecimal.fromString("0.2");
    defer b.obj.decref();

    // Precise addition: 0.1 + 0.2 = 0.3 (no floating point errors!)
    const sum = try a.add(b);
    defer sum.obj.decref();

    const expected = try py.PyDecimal.fromString("0.3");
    defer expected.obj.decref();

    const is_equal = try sum.eq(expected);
    std.debug.print("0.1 + 0.2 == 0.3: {}\n", .{is_equal});

    // Financial calculations
    const price = try py.PyDecimal.fromString("19.99");
    defer price.obj.decref();

    const quantity = try py.PyDecimal.fromInt(3);
    defer quantity.obj.decref();

    const total = try price.mul(quantity);
    defer total.obj.decref();

    // Round to 2 decimal places
    const rounded = try total.round(2);

    return rounded.obj;
}

// ============================================================================
// PyPath Examples
// ============================================================================

/// Demonstrate path operations
pub fn path_operations() !py.PyObject {
    // Get current working directory
    const cwd = try py.PyPath.cwd();
    defer cwd.obj.decref();

    // Join paths
    const subdir = try cwd.joinPath("example");
    defer subdir.obj.decref();

    // Get parent
    const parent = try subdir.parent();
    defer parent.obj.decref();

    // Check existence
    const exists = try cwd.exists();
    std.debug.print("CWD exists: {}\n", .{exists});

    // Get home directory
    const home = try py.PyPath.home();
    defer home.obj.decref();

    // Convert to string
    const path_str = try home.toString();

    return path_str.obj;
}

// ============================================================================
// PyUUID Examples
// ============================================================================

/// Demonstrate UUID generation and manipulation
pub fn uuid_operations() !py.PyObject {
    // Generate random UUID (version 4)
    const uuid = try py.PyUUID.uuid4();
    defer uuid.obj.decref();

    // Convert to string
    const uuid_str = try uuid.toString();
    defer uuid_str.obj.decref();

    const str_val = try py.PyString.asSlice(uuid_str.obj);
    defer str_val.decref();
    std.debug.print("Random UUID: {s}\n", .{str_val.buf});

    // Get hex representation
    const hex = try uuid.toHex();
    defer hex.obj.decref();

    // Generate namespace-based UUID
    const ns = try py.PyUUID.namespaceDNS();
    defer ns.obj.decref();

    const uuid5 = try py.PyUUID.uuid5(ns, "example.com");
    defer uuid5.obj.decref();

    // Get version
    const version = try uuid5.version();
    if (version) |v| {
        std.debug.print("UUID version: {}\n", .{v});
    }

    uuid.obj.incref();
    return uuid.obj;
}

// ============================================================================
// Integration Example
// ============================================================================

/// Demonstrate using multiple types together
pub fn integration_example() !py.PyObject {
    // Create a report with current datetime, decimal calculations, and UUID
    const now = try py.PyDateTime.now();
    defer now.obj.decref();

    const report_id = try py.PyUUID.uuid4();
    defer report_id.obj.decref();

    const total = try py.PyDecimal.fromString("12345.67");
    defer total.obj.decref();

    const tax_rate = try py.PyDecimal.fromString("0.08");
    defer tax_rate.obj.decref();

    const tax = try total.mul(tax_rate);
    defer tax.obj.decref();

    const grand_total = try total.add(tax);
    defer grand_total.obj.decref();

    // Create a dict to return
    const result = try py.dict(root);

    const date_key = try py.create(root, "date");
    defer date_key.decref();
    _ = try result.callMethod("__setitem__", .{ date_key, now.obj });

    const id_key = try py.create(root, "id");
    defer id_key.decref();
    _ = try result.callMethod("__setitem__", .{ id_key, report_id.obj });

    const total_key = try py.create(root, "total");
    defer total_key.decref();
    _ = try result.callMethod("__setitem__", .{ total_key, grand_total.obj });

    return result;
}

comptime {
    py.rootmodule(root);
}
