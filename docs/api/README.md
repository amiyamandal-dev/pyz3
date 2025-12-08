# pyz3 API Reference

Complete API documentation for the pyz3 (ziggy-pydust) framework.

## Table of Contents

- [Core API](#core-api)
- [Type System](#type-system)
- [Performance Optimizations](#performance-optimizations)
- [Memory Management](#memory-management)
- [Error Handling](#error-handling)
- [Advanced Topics](#advanced-topics)

## Overview

pyz3 is a high-performance framework for building Python extension modules in Zig. It provides:

- **Automatic type conversion** between Zig and Python
- **Zero-copy NumPy integration** (when enabled)
- **Memory-safe** Python extensions leveraging Zig's safety features
- **High performance** with GIL caching, fast paths, and object pooling
- **Complete CLI toolkit** for development, building, and testing

## Quick Start

### Basic Module

```zig
const py = @import("pyz3");

pub fn hello() []const u8 {
    return "Hello from Zig!";
}

pub fn add(args: struct { a: i64, b: i64 }) i64 {
    return args.a + args.b;
}

comptime {
    py.rootmodule(@This());
}
```

### Building

```bash
pyz3 build
pyz3 develop  # Install in development mode
pyz3 test    # Run tests
```

## Core API

### Module Registration

#### `py.rootmodule(comptime definition: type)`

Register a Zig module as a Python module.

**Parameters:**
- `definition`: The type containing your module's functions and classes

**Example:**
```zig
comptime {
    py.rootmodule(@This());
}
```

#### `py.module(comptime definition: type) Definition`

Register a submodule within an existing module.

**Returns:** A `Definition` that can be assigned to a struct field

**Example:**
```zig
pub const submod = py.module(struct {
    pub fn func() void {}
});
```

### Class Registration

#### `py.class(comptime definition: type) Definition`

Register a Zig struct as a Python class.

**Example:**
```zig
pub const MyClass = py.class(struct {
    value: i64,

    pub fn __init__(args: struct { value: i64 }) !@This() {
        return .{ .value = args.value };
    }

    pub fn get_value(self: *const @This()) i64 {
        return self.value;
    }
});
```

### Function Definitions

Functions are automatically discovered and exported to Python. They can have various signatures:

#### No Arguments

```zig
pub fn simple_function() []const u8 {
    return "Hello!";
}
```

#### With Arguments

```zig
pub fn with_args(args: struct { x: i64, y: i64 }) i64 {
    return args.x + args.y;
}
```

#### With Keyword Arguments

```zig
pub fn with_kwargs(args: struct {
    x: i64,
    y: i64 = 0,  // Default value makes this a kwarg
}) i64 {
    return args.x + args.y;
}
```

#### With Variadic Args

```zig
pub fn variadic(args: struct {
    x: i64,
    rest: py.Args(),  // Captures remaining positional args
}) i64 {
    var sum = x;
    for (rest) |arg| {
        sum += try py.as(@This(), i64, arg);
    }
    return sum;
}
```

#### Error Handling

```zig
pub fn may_fail(args: struct { value: i64 }) !i64 {
    if (args.value < 0) {
        return error.NegativeValue;
    }
    return args.value * 2;
}
```

## Type System

### Primitive Types

| Zig Type | Python Type | Fast Path | Pooled |
|----------|-------------|-----------|---------|
| `bool` | `bool` | ✅ | ✅ (True/False) |
| `i8`, `i16`, `i32`, `i64` | `int` | ✅ (i64) | ✅ (-5 to 256) |
| `u8`, `u16`, `u32`, `u64` | `int` | ❌ | ❌ |
| `f16`, `f32`, `f64` | `float` | ✅ (f64) | ❌ |
| `[]const u8` | `str` | ✅ | ❌ |
| `void` | `None` | ✅ | ✅ |

### Container Types

#### Tuples

```zig
pub fn return_tuple() struct { i64, []const u8 } {
    return .{ 42, "hello" };
}
```

Python:
```python
>>> mod.return_tuple()
(42, 'hello')
```

#### Dictionaries

```zig
pub fn return_dict() struct { x: i64, y: []const u8 } {
    return .{ .x = 42, .y = "hello" };
}
```

Python:
```python
>>> mod.return_dict()
{'x': 42, 'y': 'hello'}
```

#### Lists

```zig
pub fn return_list() !py.PyList {
    const list = try py.PyList.new();
    try list.append((try py.PyLong.create(1)).obj);
    try list.append((try py.PyLong.create(2)).obj);
    return list;
}
```

### Optional Types

```zig
pub fn maybe_value(args: struct { flag: bool }) ?i64 {
    return if (args.flag) 42 else null;
}
```

Python:
```python
>>> mod.maybe_value(True)
42
>>> mod.maybe_value(False)
None
```

### Python Type Wrappers

All Python types have corresponding Zig wrappers:

- `py.PyObject` - Generic Python object
- `py.PyString` - str
- `py.PyLong` - int
- `py.PyFloat` - float
- `py.PyBool` - bool
- `py.PyList` - list
- `py.PyDict(root)` - dict
- `py.PyTuple(root)` - tuple
- `py.PySet` - set
- ... and 40+ more types

## Performance Optimizations

### GIL State Caching

**Optimization:** Tracks GIL acquisition state to avoid redundant acquire/release calls

**Performance Gain:** 10-100x for nested allocations

**How It Works:**
- Thread-local variable tracks GIL depth
- Only acquires GIL if not already held
- Automatic in all memory allocations

**Example Impact:**
```zig
pub fn nested_alloc() !void {
    // GIL acquired once for all allocations
    const buf1 = try py.allocator.alloc(u8, 1024);  // Acquires GIL
    defer py.allocator.free(buf1);

    const buf2 = try py.allocator.alloc(u8, 2048);  // Reuses GIL
    defer py.allocator.free(buf2);

    const buf3 = try py.allocator.alloc(u8, 512);   // Reuses GIL
    defer py.allocator.free(buf3);
    // GIL released once at end
}
```

### Fast Paths for Primitives

**Optimization:** Direct FFI calls for common types, bypassing generic trampolines

**Performance Gain:** 2-5x for primitive conversions

**Optimized Types:**
- `i64`, `i32`, `i16`, `i8` → `PyLong_FromLongLong`
- `f64` → `PyFloat_FromDouble`
- `bool` → Returns cached True/False
- `[]const u8` → `PyUnicode_FromStringAndSize`

**Example:**
```zig
// Fast path automatically used
pub fn fast_int(args: struct { value: i64 }) i64 {
    return args.value * 2;  // Direct PyLong_FromLongLong call
}
```

### Object Pooling

**Optimization:** Caches frequently used Python objects

**Performance Gain:** 1.5-3x for small integer operations

**Pooled Objects:**
- Small integers: -5 to 256 (same as CPython)
- Boolean values: True, False
- None singleton
- Empty tuple, dict, list (when applicable)

**Example:**
```zig
pub fn pooled_int(args: struct { value: i64 }) i64 {
    // If value is -5 to 256, uses cached object
    return args.value;
}
```

## Memory Management

### Allocator

Use `py.allocator` for all Python-managed memory:

```zig
const buffer = try py.allocator.alloc(u8, 1024);
defer py.allocator.free(buffer);
```

### Reference Counting

#### Manual Reference Counting

```zig
const obj = try py.PyString.create("hello");
obj.obj.incref();  // Increment reference count
defer obj.obj.decref();  // Decrement reference count
```

#### Ownership Rules

- Functions that create objects return **new references**
- Functions that return existing objects may return **borrowed references**
- Always `incref()` if you need to keep a borrowed reference
- Always `decref()` when done with an owned reference

### Example: Safe Object Handling

```zig
pub fn process_string(args: struct { text: []const u8 }) !py.PyString {
    // Create new object (new reference)
    const str_obj = try py.PyString.create(args.text);

    // If returning, caller owns the reference
    return str_obj;

    // If not returning, must decref
    // defer str_obj.obj.decref();
}
```

## Error Handling

### Error Types

```zig
pub const PyError = error{
    PyRaised,     // Exception already set in Python
    OutOfMemory,  // Memory allocation failed
} || Allocator.Error;
```

### Raising Exceptions

```zig
pub fn validate(args: struct { value: i64 }) !i64 {
    if (args.value < 0) {
        return py.ValueError(@This()).raise("Value must be non-negative");
    }
    return args.value;
}
```

### Available Exception Types

All Python exceptions are available:
- `py.ValueError(root)`
- `py.TypeError(root)`
- `py.RuntimeError(root)`
- `py.IndexError(root)`
- `py.KeyError(root)`
- ... and all other standard exceptions

### Custom Error Messages

```zig
return py.RuntimeError(@This()).raiseFmt(
    "Expected {d} items, got {d}",
    .{ expected, actual }
);
```

## Advanced Topics

### Initialization and Finalization

```zig
test "using Python" {
    py.initialize();  // Start Python interpreter
    defer py.finalize();  // Clean up

    // Your test code here
}
```

### Calling Python from Zig

```zig
pub fn call_python() !py.PyObject {
    // Import a module
    const math_mod = try py.import("math");
    defer math_mod.decref();

    // Get a function
    const sqrt_fn = try math_mod.getAttr("sqrt");
    defer sqrt_fn.decref();

    // Call it
    const arg = try py.PyFloat.create(16.0);
    defer arg.obj.decref();

    const result = try py.call(sqrt_fn, .{arg.obj}, .{});
    // Result is owned by caller
    return result;
}
```

### Working with the GIL

```zig
// Acquire GIL
const gil = py.gil();
defer gil.release();

// Or release GIL for I/O operations
const no_gil = py.nogil();
defer no_gil.reacquire();
```

### Type Conversion

#### Zig → Python

```zig
const py_obj = try py.create(@This(), my_zig_value);
defer py_obj.decref();
```

#### Python → Zig

```zig
const zig_value = try py.as(@This(), i64, py_obj);
```

#### Unchecked Conversions (for performance)

```zig
// Skip type checking (faster but unsafe)
const zig_value = py.unchecked(@This(), i64, py_obj);
```

## Best Practices

1. **Always use `defer` for cleanup**
   ```zig
   const obj = try create_object();
   defer obj.decref();
   ```

2. **Prefer `py.allocator` over std.heap**
   - Integrates with Python's memory system
   - GIL caching optimization
   - Memory leak detection in tests

3. **Use error unions, not panics**
   ```zig
   // Good
   pub fn safe_div(a: i64, b: i64) !i64 {
       if (b == 0) return error.DivisionByZero;
       return @divTrunc(a, b);
   }

   // Bad
   pub fn unsafe_div(a: i64, b: i64) i64 {
       return @divTrunc(a, b);  // Panics on b=0
   }
   ```

4. **Leverage fast paths for hot code**
   - Use `i64` instead of `i32` for most integers
   - Use `f64` instead of `f32` for floats
   - Keep strings as `[]const u8`

5. **Profile before optimizing**
   ```bash
   pytest tests/benchmark_optimizations.py -v -s
   ```

## Performance Tips

- **GIL Caching**: Automatically optimizes nested allocations
- **Fast Paths**: Use `i64`, `f64`, `bool`, `[]const u8` for best performance
- **Object Pooling**: Small integers (-5 to 256) are cached
- **Avoid Copies**: Return slices instead of copying data when possible
- **Batch Operations**: Process multiple items in Zig rather than Python

## See Also

- [Type Conversion Guide](type-conversion.md)
- [Memory Management Guide](memory.md)
- [Performance Optimization Guide](performance.md)
- [Error Handling Guide](errors.md)
- [Examples](../example/)
