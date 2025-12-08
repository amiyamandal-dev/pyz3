# Type Conversion Guide

Complete guide to type conversion between Zig and Python in pyz3.

## Overview

pyz3 automatically converts between Zig and Python types using compile-time introspection. This guide covers all supported conversions and their behavior.

## Automatic Conversions

### Function Arguments

Arguments are automatically converted from Python to Zig:

```zig
pub fn greet(args: struct { name: []const u8, age: i64 }) []const u8 {
    // args.name is automatically converted from Python str to []const u8
    // args.age is automatically converted from Python int to i64
    return "Hello!";
}
```

Python usage:
```python
>>> module.greet(name="Alice", age=30)
"Hello!"
```

### Return Values

Return values are automatically converted from Zig to Python:

```zig
pub fn get_info() struct { name: []const u8, count: i64 } {
    return .{ .name = "example", .count = 42 };
}
```

Python usage:
```python
>>> module.get_info()
{'name': 'example', 'count': 42}
```

## Primitive Type Conversions

### Integers

| Zig Type | Python Type | Notes |
|----------|-------------|-------|
| `i8`, `i16`, `i32`, `i64` | `int` | Signed integers |
| `u8`, `u16`, `u32`, `u64` | `int` | Unsigned integers |
| `isize`, `usize` | `int` | Platform-specific |
| `comptime_int` | `int` | Compile-time constant |

**Example:**
```zig
pub fn integer_types(args: struct {
    small: i8,
    medium: i32,
    large: i64,
    unsigned: u64,
}) struct { i8, i32, i64, u64 } {
    return .{ args.small, args.medium, args.large, args.unsigned };
}
```

**Best Practice:** Use `i64` for most integers to leverage fast path optimization.

### Floating Point

| Zig Type | Python Type | Notes |
|----------|-------------|-------|
| `f16`, `f32`, `f64` | `float` | IEEE 754 floats |
| `comptime_float` | `float` | Compile-time constant |

**Example:**
```zig
pub fn float_types(args: struct {
    precise: f64,
    approx: f32,
}) f64 {
    return args.precise + @as(f64, args.approx);
}
```

**Best Practice:** Use `f64` to leverage fast path optimization.

### Boolean

| Zig Type | Python Type | Notes |
|----------|-------------|-------|
| `bool` | `bool` | True/False values |

**Example:**
```zig
pub fn bool_ops(args: struct { flag: bool }) bool {
    return !args.flag;
}
```

**Performance:** Booleans use cached True/False objects (fast path).

### Strings

| Zig Type | Python Type | Notes |
|----------|-------------|-------|
| `[]const u8` | `str` | UTF-8 encoded |
| `[]u8` | Not supported | Use `[]const u8` |
| `[N]u8` | `str` | Fixed-size array |

**Example:**
```zig
pub fn string_ops(args: struct {
    text: []const u8,
}) []const u8 {
    return args.text;
}
```

**Python:**
```python
>>> module.string_ops(text="Hello, 世界!")
"Hello, 世界!"
```

**Performance:** String conversion uses fast path for `[]const u8`.

### Void and None

| Zig Type | Python Type | Notes |
|----------|-------------|-------|
| `void` | `None` | No value |

**Example:**
```zig
pub fn no_return(args: struct { value: i64 }) void {
    // Side effects only
}
```

**Python:**
```python
>>> result = module.no_return(value=42)
>>> print(result)
None
```

## Container Type Conversions

### Tuples

Zig tuple structs convert to Python tuples:

```zig
pub fn make_tuple() struct { i64, []const u8, bool } {
    return .{ 42, "hello", true };
}
```

**Python:**
```python
>>> module.make_tuple()
(42, 'hello', True)
```

**Accessing tuple elements:**
```zig
pub fn use_tuple(args: struct {
    data: struct { i64, []const u8 }
}) i64 {
    const num = args.data[0];
    const text = args.data[1];
    return num;
}
```

### Dictionaries

Zig named structs convert to Python dicts:

```zig
pub fn make_dict() struct { x: i64, y: i64, name: []const u8 } {
    return .{ .x = 10, .y = 20, .name = "point" };
}
```

**Python:**
```python
>>> module.make_dict()
{'x': 10, 'y': 20, 'name': 'point'}
```

**Nested structures:**
```zig
pub fn nested_dict() struct {
    outer: struct {
        inner: struct { value: i64 }
    }
} {
    return .{ .outer = .{ .inner = .{ .value = 42 } } };
}
```

**Python:**
```python
>>> module.nested_dict()
{'outer': {'inner': {'value': 42}}}
```

### Lists

Use `py.PyList` for Python lists:

```zig
pub fn make_list() !py.PyList {
    const list = try py.PyList.new();

    try list.append((try py.PyLong.create(1)).obj);
    try list.append((try py.PyLong.create(2)).obj);
    try list.append((try py.PyLong.create(3)).obj);

    return list;
}
```

**Python:**
```python
>>> module.make_list()
[1, 2, 3]
```

**Processing lists:**
```zig
pub fn process_list(args: struct { items: py.PyList }) !i64 {
    var sum: i64 = 0;
    for (0..args.items.length()) |i| {
        const item = try args.items.getItem(py.PyLong, i);
        sum += try item.as(i64);
    }
    return sum;
}
```

### Sets and FrozenSets

```zig
pub fn make_set() !py.PySet {
    const set = try py.PySet.new();

    try set.add((try py.PyLong.create(1)).obj);
    try set.add((try py.PyLong.create(2)).obj);
    try set.add((try py.PyLong.create(1)).obj);  // Duplicate ignored

    return set;
}
```

**Python:**
```python
>>> module.make_set()
{1, 2}
```

## Optional Type Conversions

### Optional Values

Zig optionals convert to Python None or value:

```zig
pub fn maybe_value(args: struct { flag: bool }) ?i64 {
    return if (args.flag) 42 else null;
}
```

**Python:**
```python
>>> module.maybe_value(flag=True)
42
>>> module.maybe_value(flag=False)
None
```

### Optional Arguments

Use default values for optional parameters:

```zig
pub fn with_optional(args: struct {
    required: i64,
    optional: i64 = 100,  // Default value
}) i64 {
    return args.required + args.optional;
}
```

**Python:**
```python
>>> module.with_optional(required=10)
110
>>> module.with_optional(required=10, optional=5)
15
```

### Nested Optionals

```zig
pub fn nested_optional(args: struct {
    value: ?struct { x: i64, y: i64 }
}) ?i64 {
    if (args.value) |v| {
        return v.x + v.y;
    }
    return null;
}
```

## Error Union Conversions

### Error Handling

Zig errors convert to Python exceptions:

```zig
pub fn may_fail(args: struct { value: i64 }) !i64 {
    if (args.value < 0) {
        return error.NegativeValue;
    }
    if (args.value == 0) {
        return py.ValueError(@This()).raise("Value cannot be zero");
    }
    return args.value * 2;
}
```

**Python:**
```python
>>> module.may_fail(value=5)
10
>>> module.may_fail(value=-1)
RuntimeError: NegativeValue
>>> module.may_fail(value=0)
ValueError: Value cannot be zero
```

### Custom Error Messages

```zig
pub fn validate_range(args: struct { value: i64, min: i64, max: i64 }) !i64 {
    if (args.value < args.min or args.value > args.max) {
        return py.ValueError(@This()).raiseFmt(
            "Value {d} outside range [{d}, {d}]",
            .{ args.value, args.min, args.max }
        );
    }
    return args.value;
}
```

### Error + Optional

```zig
pub fn complex_return(args: struct { flag: bool }) !?i64 {
    if (args.flag) {
        return 42;
    }
    return null;
}
```

## Python Object Wrappers

### Using Python Types in Zig

All Python types have Zig wrappers:

```zig
pub fn use_python_types(args: struct {
    text: py.PyString,
    number: py.PyLong,
    container: py.PyList,
}) !py.PyDict(@This()) {
    const dict = try py.PyDict(@This()).new();

    try dict.setItem(try py.PyString.create("text"), args.text.obj);
    try dict.setItem(try py.PyString.create("number"), args.number.obj);
    try dict.setItem(try py.PyString.create("container"), args.container.obj);

    return dict;
}
```

### Creating Python Objects

```zig
// Strings
const str = try py.PyString.create("hello");
defer str.obj.decref();

// Integers
const num = try py.PyLong.create(42);
defer num.obj.decref();

// Floats
const float = try py.PyFloat.create(3.14);
defer float.obj.decref();

// Lists
const list = try py.PyList.new();
defer list.obj.decref();

// Dicts
const dict = try py.PyDict(@This()).new();
defer dict.obj.decref();

// Tuples
const tuple = try py.PyTuple(@This()).create(.{ 1, "hello", true });
defer tuple.obj.decref();
```

### Converting Python Objects to Zig

```zig
pub fn extract_value(args: struct { obj: py.PyObject }) !i64 {
    // Convert PyObject to i64
    const num = try py.as(@This(), i64, args.obj);
    return num;
}
```

## Manual Conversion Functions

### `py.create()` - Zig to Python

```zig
pub fn manual_create() !py.PyObject {
    // Manual conversion
    const obj = try py.create(@This(), 42);
    return obj;  // Caller owns reference
}
```

### `py.as()` - Python to Zig

```zig
pub fn manual_extract(args: struct { obj: py.PyObject }) !i64 {
    // Manual conversion with type checking
    const value = try py.as(@This(), i64, args.obj);
    return value;
}
```

### `py.unchecked()` - Fast Unchecked Conversion

```zig
pub fn unchecked_extract(args: struct { obj: py.PyObject }) i64 {
    // No type checking - faster but unsafe
    const value = py.unchecked(@This(), i64, args.obj);
    return value;
}
```

**Warning:** Only use `unchecked()` when you're certain of the type!

## Advanced Conversions

### Variadic Arguments

```zig
pub fn sum_all(args: struct {
    first: i64,
    rest: py.Args(),  // Variable positional args
}) !i64 {
    var total = args.first;
    for (args.rest) |arg| {
        const value = try py.as(@This(), i64, arg);
        total += value;
    }
    return total;
}
```

**Python:**
```python
>>> module.sum_all(1, 2, 3, 4, 5)
15
```

### Keyword Arguments

```zig
pub fn with_kwargs(args: struct {
    required: i64,
    kwargs: py.Kwargs(),  // Variable keyword args
}) !py.PyDict(@This()) {
    const result = try py.PyDict(@This()).new();

    try result.setItem(
        try py.PyString.create("required"),
        try py.create(@This(), args.required)
    );

    var iter = args.kwargs.iterator();
    while (iter.next()) |entry| {
        try result.setItem(
            try py.PyString.create(entry.key_ptr.*),
            entry.value_ptr.*
        );
    }

    return result;
}
```

**Python:**
```python
>>> module.with_kwargs(required=1, a=2, b=3, c=4)
{'required': 1, 'a': 2, 'b': 3, 'c': 4}
```

### Mixed Args and Kwargs

```zig
pub fn mixed(args: struct {
    pos1: i64,
    pos2: i64 = 0,
    pos_rest: py.Args(),
    kw_rest: py.Kwargs(),
}) !i64 {
    var sum = args.pos1 + args.pos2;

    for (args.pos_rest) |arg| {
        sum += try py.as(@This(), i64, arg);
    }

    var iter = args.kw_rest.valueIterator();
    while (iter.next()) |value| {
        sum += try py.as(@This(), i64, value.*);
    }

    return sum;
}
```

## Type Checking

### Runtime Type Checking

```zig
pub fn check_type(args: struct { obj: py.PyObject }) ![]const u8 {
    if (ffi.PyLong_Check(args.obj.py) != 0) {
        return "integer";
    } else if (ffi.PyFloat_Check(args.obj.py) != 0) {
        return "float";
    } else if (ffi.PyUnicode_Check(args.obj.py) != 0) {
        return "string";
    } else if (ffi.PyList_Check(args.obj.py) != 0) {
        return "list";
    } else {
        return "unknown";
    }
}
```

### isinstance Check

```zig
pub fn check_instance(args: struct {
    obj: py.PyObject,
    cls: py.PyObject,
}) !bool {
    return try py.isinstance(@This(), args.obj, args.cls);
}
```

## Performance Considerations

### Fast Path Types (Recommended)

Use these types for best performance:
- `i64` for integers
- `f64` for floats
- `bool` for booleans
- `[]const u8` for strings

### Slow Path Types (Avoid in Hot Code)

These types use generic trampolines:
- `i32`, `i16`, `i8` (use `i64` instead)
- `u32`, `u16`, `u8` (convert to `i64`)
- `f32`, `f16` (use `f64` instead)

### Object Pooling Range

Small integers (-5 to 256) are cached:

```zig
// Fast - uses pooled object
pub fn fast(args: struct { x: i64 }) i64 {
    return 42;  // Pooled
}

// Slower - creates new object
pub fn slower(args: struct { x: i64 }) i64 {
    return 1000;  // Not pooled
}
```

### Conversion Overhead

**Lowest overhead:**
```zig
pub fn lowest() void {}  // No conversion
```

**Low overhead:**
```zig
pub fn low(args: struct { x: i64 }) i64 {  // Fast path
    return args.x;
}
```

**Medium overhead:**
```zig
pub fn medium(args: struct {  // Multiple fast paths
    a: i64, b: f64, c: bool
}) struct { i64, f64, bool } {
    return .{ args.a, args.b, args.c };
}
```

**Higher overhead:**
```zig
pub fn higher(args: struct {
    data: struct { x: i64, y: i64 }  // Nested struct
}) struct { i64, i64 } {
    return .{ args.data.x, args.data.y };
}
```

## Common Patterns

### Pattern: Return Multiple Values

```zig
pub fn divide_with_remainder(args: struct {
    dividend: i64,
    divisor: i64
}) !struct { quotient: i64, remainder: i64 } {
    if (args.divisor == 0) {
        return py.ValueError(@This()).raise("Division by zero");
    }
    return .{
        .quotient = @divTrunc(args.dividend, args.divisor),
        .remainder = @mod(args.dividend, args.divisor),
    };
}
```

### Pattern: Builder Style

```zig
pub fn build_config(args: struct {
    host: []const u8 = "localhost",
    port: i64 = 8080,
    debug: bool = false,
}) struct { host: []const u8, port: i64, debug: bool } {
    return .{
        .host = args.host,
        .port = args.port,
        .debug = args.debug,
    };
}
```

### Pattern: Type Polymorphism

```zig
pub fn stringify(args: struct { value: py.PyObject }) ![]const u8 {
    const str_obj = try py.str(@This(), args.value);
    defer str_obj.decref();

    const str_type = try py.PyString.from.checked(@This(), str_obj);
    return str_type.asSlice();
}
```

## See Also

- [API Reference](README.md)
- [Performance Guide](performance.md)
- [Memory Management](memory.md)
- [Error Handling](errors.md)
