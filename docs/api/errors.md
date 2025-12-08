# Error Handling Guide

Complete guide to error handling in pyz3, including Python exceptions, Zig errors, and best practices.

## Overview

pyz3 bridges two error systems:
1. **Zig errors** - Compile-time error sets and error unions
2. **Python exceptions** - Runtime exception objects

Understanding how these interact is crucial for robust extensions.

## Error Types

### PyError

The main error type for pyz3:

```zig
pub const PyError = error{
    PyRaised,     // Python exception already set
    OutOfMemory,  // Memory allocation failed
} || Allocator.Error;
```

### Using PyError

```zig
pub fn may_fail() PyError!i64 {
    const obj = try py.PyLong.create(42);
    defer obj.obj.decref();

    return try obj.as(i64);
}
```

## Raising Python Exceptions

### Built-in Exception Types

All standard Python exceptions are available:

```zig
// ValueError
return py.ValueError(@This()).raise("Invalid value");

// TypeError
return py.TypeError(@This()).raise("Wrong type");

// RuntimeError
return py.RuntimeError(@This()).raise("Runtime error occurred");

// IndexError
return py.IndexError(@This()).raise("Index out of bounds");

// KeyError
return py.KeyError(@This()).raise("Key not found");

// AttributeError
return py.AttributeError(@This()).raise("Attribute missing");

// NotImplementedError
return py.NotImplementedError(@This()).raise("Not implemented");

// IOError / OSError
return py.IOError(@This()).raise("I/O error");

// MemoryError
return py.MemoryError(@This()).raise("Out of memory");

// ZeroDivisionError
return py.ZeroDivisionError(@This()).raise("Division by zero");

// StopIteration
return py.StopIteration(@This()).raise("Iterator exhausted");

// And many more...
```

### Basic Exception Raising

```zig
pub fn validate_positive(args: struct { value: i64 }) !i64 {
    if (args.value < 0) {
        return py.ValueError(@This()).raise("Value must be non-negative");
    }
    return args.value;
}
```

Python usage:
```python
>>> module.validate_positive(value=5)
5
>>> module.validate_positive(value=-1)
ValueError: Value must be non-negative
```

### Formatted Exception Messages

```zig
pub fn validate_range(args: struct {
    value: i64,
    min: i64,
    max: i64
}) !i64 {
    if (args.value < args.min or args.value > args.max) {
        return py.ValueError(@This()).raiseFmt(
            "Value {d} outside range [{d}, {d}]",
            .{ args.value, args.min, args.max }
        );
    }
    return args.value;
}
```

Python usage:
```python
>>> module.validate_range(value=150, min=0, max=100)
ValueError: Value 150 outside range [0, 100]
```

## Converting Zig Errors to Python Exceptions

### Automatic Conversion

Zig errors are automatically converted to Python RuntimeError:

```zig
pub fn zig_error_example(args: struct { value: i64 }) !i64 {
    if (args.value == 0) {
        return error.ZeroNotAllowed;  // Becomes RuntimeError: ZeroNotAllowed
    }
    return args.value;
}
```

Python usage:
```python
>>> module.zig_error_example(value=5)
5
>>> module.zig_error_example(value=0)
RuntimeError: ZeroNotAllowed
```

### Custom Error Messages

Map Zig errors to specific Python exceptions:

```zig
pub const ValidationError = error{
    TooSmall,
    TooLarge,
    InvalidFormat,
};

pub fn validate(args: struct { value: i64 }) !i64 {
    if (args.value < 0) {
        return py.ValueError(@This()).raise("Value too small");
    }
    if (args.value > 100) {
        return py.ValueError(@This()).raise("Value too large");
    }
    return args.value;
}
```

### Error Mapping

```zig
fn mapError(err: anyerror) PyError!void {
    return switch (err) {
        error.OutOfMemory => py.MemoryError(@This()).raise("Out of memory"),
        error.FileNotFound => py.FileNotFoundError(@This()).raise("File not found"),
        error.PermissionDenied => py.PermissionError(@This()).raise("Permission denied"),
        error.InvalidFormat => py.ValueError(@This()).raise("Invalid format"),
        else => py.RuntimeError(@This()).raiseFmt("Error: {s}", .{@errorName(err)}),
    };
}

pub fn safe_operation(args: struct { path: []const u8 }) !void {
    const result = performOperation(args.path) catch |err| {
        try mapError(err);
        return;
    };
    _ = result;
}
```

## Catching and Handling Exceptions

### Check for Python Exception

```zig
pub fn check_exception() !bool {
    // Call Python function that might raise
    const math_mod = try py.import("math");
    defer math_mod.decref();

    const sqrt = try math_mod.getAttr("sqrt");
    defer sqrt.decref();

    // This will raise ValueError
    const result = py.call(sqrt, .{
        (try py.PyFloat.create(-1.0)).obj
    }, .{}) catch |err| {
        if (err == PyError.PyRaised) {
            // Exception was raised
            return true;
        }
        return err;
    };

    result.decref();
    return false;
}
```

### Getting Exception Info

```zig
const ffi = @import("ffi");

pub fn get_exception_info() !struct { type: []const u8, message: []const u8 } {
    // Fetch current exception
    var ptype: ?*ffi.PyObject = null;
    var pvalue: ?*ffi.PyObject = null;
    var ptraceback: ?*ffi.PyObject = null;

    ffi.PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    if (ptype == null) {
        return error.NoException;
    }

    defer {
        if (ptype) |t| ffi.Py_DecRef(t);
        if (pvalue) |v| ffi.Py_DecRef(v);
        if (ptraceback) |tb| ffi.Py_DecRef(tb);
    }

    // Get exception type name
    const type_obj = py.PyObject{ .py = ptype.? };
    const type_name = try type_obj.getTypeName();

    // Get exception message
    const value_obj = if (pvalue) |v| py.PyObject{ .py = v } else type_obj;
    const msg_obj = try py.str(@This(), value_obj);
    defer msg_obj.decref();

    const msg_str = try py.PyString.from.checked(@This(), msg_obj);
    const message = msg_str.asSlice();

    return .{ .type = type_name, .message = message };
}
```

### Clearing Exceptions

```zig
pub fn clear_exception() void {
    const ffi = @import("ffi");
    ffi.PyErr_Clear();
}
```

## Error Patterns

### Pattern: Validation

```zig
pub fn validate_email(args: struct { email: []const u8 }) ![]const u8 {
    // Check for @ symbol
    const has_at = std.mem.indexOf(u8, args.email, "@") != null;
    if (!has_at) {
        return py.ValueError(@This()).raise("Email must contain @");
    }

    // Check for domain
    const at_idx = std.mem.indexOf(u8, args.email, "@").?;
    if (at_idx + 1 >= args.email.len) {
        return py.ValueError(@This()).raise("Email must have domain");
    }

    return args.email;
}
```

### Pattern: Resource Cleanup on Error

```zig
pub fn safe_file_operation(args: struct { path: []const u8 }) !void {
    const file = std.fs.cwd().openFile(args.path, .{}) catch |err| {
        return py.IOError(@This()).raiseFmt(
            "Failed to open {s}: {s}",
            .{ args.path, @errorName(err) }
        );
    };
    defer file.close();  // Always closes, even on error

    // Work with file...
    const buffer = try py.allocator.alloc(u8, 1024);
    defer py.allocator.free(buffer);  // Always frees

    _ = try file.read(buffer);
}
```

### Pattern: Nested Errors

```zig
pub fn nested_operation(args: struct { value: i64 }) !i64 {
    // Inner operation that might fail
    const inner = inner_operation(args.value) catch |err| {
        return py.RuntimeError(@This()).raiseFmt(
            "Inner operation failed: {s}",
            .{@errorName(err)}
        );
    };

    // Outer operation
    if (inner < 0) {
        return py.ValueError(@This()).raise("Result must be non-negative");
    }

    return inner;
}

fn inner_operation(value: i64) !i64 {
    if (value == 0) {
        return error.DivisionByZero;
    }
    return @divTrunc(100, value);
}
```

### Pattern: Optional with Error

```zig
pub fn find_item(args: struct {
    items: py.PyList,
    target: i64
}) !?i64 {
    for (0..args.items.length()) |i| {
        const item = try args.items.getItem(py.PyLong, i);
        const value = try item.as(i64);

        if (value == args.target) {
            return i;  // Found at index i
        }
    }

    return null;  // Not found
}
```

## Error Chaining

### Python 3.x Exception Chaining

```zig
pub fn chained_error() !void {
    const inner_err = inner_op() catch |err| {
        // Raise new exception while preserving original
        return py.RuntimeError(@This()).raiseFmt(
            "Operation failed due to: {s}",
            .{@errorName(err)}
        );
    };
    _ = inner_err;
}
```

Python sees:
```python
RuntimeError: Operation failed due to: OutOfMemory
```

## Testing Error Conditions

### Test Exception Raising

```zig
test "validate raises on negative" {
    py.initialize();
    defer py.finalize();

    const result = validate_positive(.{ .value = -1 });

    // Expect error
    try std.testing.expectError(PyError.PyRaised, result);
}
```

### Test Error Messages

```python
def test_error_message():
    import pytest
    from mymodule import validate_range

    with pytest.raises(ValueError) as exc_info:
        validate_range(value=150, min=0, max=100)

    assert "outside range" in str(exc_info.value)
    assert "150" in str(exc_info.value)
```

## Custom Exception Types

### Define Custom Exception in Python

```python
# module/__init__.py
class ValidationError(Exception):
    """Custom validation error"""
    pass
```

### Raise from Zig

```zig
pub fn raise_custom(args: struct { value: i64 }) !void {
    if (args.value < 0) {
        // Import the module to get custom exception
        const mod = try py.import("mymodule");
        defer mod.decref();

        const exc_class = try mod.getAttr("ValidationError");
        defer exc_class.decref();

        // Raise custom exception
        const msg = try py.PyString.create("Validation failed");
        defer msg.obj.decref();

        const ffi = @import("ffi");
        ffi.PyErr_SetObject(exc_class.py, msg.obj.py);
        return PyError.PyRaised;
    }
}
```

## Best Practices

### ✅ DO: Use Specific Exception Types

```zig
// Good - specific exception type
if (index >= array.len) {
    return py.IndexError(@This()).raise("Index out of bounds");
}

// Bad - generic RuntimeError
if (index >= array.len) {
    return error.OutOfBounds;  // Becomes generic RuntimeError
}
```

### ✅ DO: Provide Helpful Error Messages

```zig
// Good - descriptive message
return py.ValueError(@This()).raiseFmt(
    "Expected array of length {d}, got {d}",
    .{ expected_len, actual_len }
);

// Bad - vague message
return py.ValueError(@This()).raise("Invalid length");
```

### ✅ DO: Clean Up on Error

```zig
pub fn cleanup_on_error() !void {
    const resource1 = try allocate1();
    defer release1(resource1);  // Always runs

    const resource2 = try allocate2();
    defer release2(resource2);  // Always runs

    try risky_operation();  // If this fails, cleanup still happens
}
```

### ✅ DO: Use Error Returns, Not Panics

```zig
// Good - returns error
pub fn safe_divide(a: i64, b: i64) !i64 {
    if (b == 0) {
        return error.DivisionByZero;
    }
    return @divTrunc(a, b);
}

// Bad - panics
pub fn unsafe_divide(a: i64, b: i64) i64 {
    return @divTrunc(a, b);  // Panics on b=0!
}
```

### ✅ DO: Document Error Conditions

```zig
/// Validates that value is within range [min, max]
///
/// Errors:
///   - ValueError: if value is outside range
///   - TypeError: if value is not an integer
pub fn validate_range(args: struct {
    value: i64,
    min: i64,
    max: i64
}) !i64 {
    if (args.value < args.min or args.value > args.max) {
        return py.ValueError(@This()).raiseFmt(
            "Value {d} outside range [{d}, {d}]",
            .{ args.value, args.min, args.max }
        );
    }
    return args.value;
}
```

### ❌ DON'T: Swallow Errors

```zig
// Bad - silently ignores errors
pub fn bad_error_handling() void {
    const result = risky_operation() catch return;
    _ = result;
}

// Good - handle or propagate
pub fn good_error_handling() !void {
    const result = risky_operation() catch |err| {
        std.debug.print("Operation failed: {}\n", .{err});
        return err;
    };
    _ = result;
}
```

### ❌ DON'T: Forget Error Context

```zig
// Bad - loses context
pub fn bad_context(path: []const u8) !void {
    const file = std.fs.cwd().openFile(path, .{}) catch {
        return error.FileFailed;  // Lost the specific error!
    };
    defer file.close();
}

// Good - preserves context
pub fn good_context(path: []const u8) !void {
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        return py.IOError(@This()).raiseFmt(
            "Failed to open {s}: {s}",
            .{ path, @errorName(err) }
        );
    };
    defer file.close();
}
```

## Error Handling Checklist

- ✅ Use specific exception types (ValueError, TypeError, etc.)
- ✅ Include helpful error messages with context
- ✅ Use formatted messages for dynamic information
- ✅ Clean up resources with `defer`
- ✅ Propagate errors with `try` or handle with `catch`
- ✅ Document error conditions in function comments
- ✅ Test error paths with unit tests
- ✅ Map Zig errors to appropriate Python exceptions
- ✅ Return errors, don't panic
- ✅ Preserve error context when re-raising

## Common Error Scenarios

### Division by Zero

```zig
pub fn divide(args: struct { a: i64, b: i64 }) !f64 {
    if (args.b == 0) {
        return py.ZeroDivisionError(@This()).raise("Cannot divide by zero");
    }
    return @as(f64, @floatFromInt(args.a)) / @as(f64, @floatFromInt(args.b));
}
```

### Index Out of Bounds

```zig
pub fn get_item(args: struct { items: py.PyList, index: i64 }) !py.PyObject {
    const idx: usize = if (args.index < 0) {
        return py.IndexError(@This()).raise("Negative indices not supported");
    } else @intCast(args.index);

    if (idx >= args.items.length()) {
        return py.IndexError(@This()).raiseFmt(
            "Index {d} out of range for list of length {d}",
            .{ args.index, args.items.length() }
        );
    }

    return try args.items.getItem(py.PyObject, idx);
}
```

### Type Mismatch

```zig
pub fn process_number(args: struct { obj: py.PyObject }) !i64 {
    if (ffi.PyLong_Check(args.obj.py) == 0) {
        const type_name = try args.obj.getTypeName();
        return py.TypeError(@This()).raiseFmt(
            "Expected int, got {s}",
            .{type_name}
        );
    }

    return try py.as(@This(), i64, args.obj);
}
```

### Missing Key

```zig
pub fn get_required(args: struct { dict: py.PyDict(@This()), key: []const u8 }) !py.PyObject {
    const key_obj = try py.PyString.create(args.key);
    defer key_obj.obj.decref();

    const value = args.dict.getItem(key_obj.obj) catch {
        return py.KeyError(@This()).raiseFmt(
            "Required key '{s}' not found",
            .{args.key}
        );
    };

    return value orelse {
        return py.KeyError(@This()).raiseFmt(
            "Required key '{s}' not found",
            .{args.key}
        );
    };
}
```

## See Also

- [API Reference](README.md)
- [Type Conversion](type-conversion.md)
- [Memory Management](memory.md)
- [Testing Guide](../guide/testing.md)
