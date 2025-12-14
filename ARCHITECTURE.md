# pyz3 Architecture Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Type System](#type-system)
4. [Memory Management](#memory-management)
5. [Build System](#build-system)
6. [Performance Optimizations](#performance-optimizations)
7. [Extension Points](#extension-points)

## Overview

pyz3 is a framework for building high-performance Python extensions in Zig. It provides automatic binding generation, memory management, and type conversion between Zig and Python.

### Architecture Layers

```
┌─────────────────────────────────────────┐
│        Python Application               │
│        (imports .so modules)            │
└────────────────┬────────────────────────┘
                 │
         CPython C API
                 │
┌────────────────┴────────────────────────┐
│       pyz3 Framework                    │
├─────────────────────────────────────────┤
│ • Discovery (compile-time reflection)  │
│ • PyTypes (class/module generation)    │
│ • Functions (signature analysis)       │
│ • Trampoline (type conversions)        │
│ • Memory (GIL + allocators)            │
└────────────────┬────────────────────────┘
                 │
┌────────────────┴────────────────────────┐
│     Python FFI (ffi.h)                  │
│     CPython C API bindings              │
└─────────────────────────────────────────┘
```

## Core Components

### 1. Discovery (`pyz3/src/discovery.zig`)

**Purpose:** Compile-time introspection of Zig types to generate Python bindings

**Key Functions:**
- `Type(root, name, definition)` - Generate PyType_Spec from Zig struct
- `getAllIdentifiers()` - Recursively discover all Python-exposed definitions
- `findDefinition()` - Lookup type definitions at compile time

**How it works:**
```zig
comptime {
    // At compile time, discovers all pub const X = py.class(...)
    for (getAllIdentifiers(root)) |id| {
        // Generate Python type metadata
        const pytype = Type(root, id.qualifiedName, id.definition);
    }
}
```

**Complexity:** O(n) where n = number of fields + decls in type hierarchy
**Optimization:** Uses 100,000 eval branch quota for complex modules

### 2. PyTypes (`pyz3/src/pytypes.zig`)

**Purpose:** Python class and module initialization

**Key Components:**
- `Slots()` - Generate method slots (tp_init, tp_call, tp_new, etc.)
- `Bases()` - Handle inheritance
- `Members()` - Create property descriptors
- `Methods()` - Build method table

**Type Creation Flow:**
```
Zig struct definition
    ↓
Discovery finds it
    ↓
PyTypes generates PyType_Spec
    ↓
PyType_FromSpec() creates Python type
    ↓
Registered in module namespace
```

### 3. Functions (`pyz3/src/functions.zig`)

**Purpose:** Wrap Zig functions for Python calling

**Features:**
- Automatic signature analysis
- Positional and keyword argument handling
- Optional parameters
- Error/exception propagation
- Operator overloading (44 operators supported)

**Wrapper Generation:**
```zig
pub fn zigFunction(args: struct { a: i64, b: ?i64 = null }) !i64 {
    // Zig implementation
}

// Auto-generated wrapper:
fn __pyz3_wrapper(self: *PyObject, args: *PyObject, kwargs: *PyObject) callconv(.C) ?*PyObject {
    // 1. Parse args/kwargs
    // 2. Convert Python → Zig
    // 3. Call zigFunction()
    // 4. Convert Zig → Python
    // 5. Handle errors
}
```

### 4. Trampoline (`pyz3/src/trampoline.zig`)

**Purpose:** High-performance type conversion between Zig and Python

**Fast Paths:**
```zig
// Primitives: 4.9x-10.2x faster than generic conversion
inline fn wrapI64(value: i64) !PyObject
inline fn wrapF64(value: f64) !PyObject
inline fn wrapBool(value: bool) PyObject
inline fn wrapString(value: []const u8) !PyObject

// Unwrap (Python → Zig)
inline fn unwrapI64(obj: PyObject) !i64
inline fn unwrapF64(obj: PyObject) !f64
// etc.
```

**Generic Trampoline:**
```zig
pub fn Trampoline(comptime root: type, comptime T: type) type {
    return struct {
        pub fn wrap(value: T) !PyObject { /* ... */ }
        pub fn unwrap(obj: PyObject) !T { /* ... */ }
    };
}
```

**Supported Types:**
- Primitives: bool, integers (i8-i128), floats (f16-f64)
- Strings: []const u8, [:0]const u8
- Containers: arrays, slices, tuples, structs
- Python types: PyObject, PyString, PyList, etc.
- Optional types: ?T
- Error unions: !T, anyerror!T

### 5. Memory Management (`pyz3/src/mem.zig`)

**Purpose:** Safe memory management with GIL awareness

**Components:**

#### GIL Tracking
```zig
threadlocal var gil_depth: u32 = 0;

pub const ScopedGIL = struct {
    pub fn acquire() ScopedGIL {
        if (gil_depth == 0) {
            state = ffi.PyGILState_Ensure();
        }
        gil_depth += 1;
        return .{ .state = state };
    }

    pub fn release(self: ScopedGIL) void {
        gil_depth -= 1;
        if (gil_depth == 0) {
            ffi.PyGILState_Release(self.state);
        }
    }
};
```

**Performance:** 10-100x faster for nested allocations (avoids redundant GIL ops)

#### Allocator
```zig
pub const PyMemAllocator = struct {
    pub fn alloc(self: @This(), len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const gil = ScopedGIL.acquire();
        defer gil.release();

        // Align memory using header
        const full_len = len + alignment;
        const mem = ffi.PyMem_RawMalloc(full_len) orelse return null;
        // ...
    }

    pub fn free(self: @This(), buf: []u8, buf_align: u8, ret_addr: usize) void {
        const gil = ScopedGIL.acquire();
        defer gil.release();

        ffi.PyMem_RawFree(original_ptr);
    }
};
```

#### Object Pool (`pyz3/src/object_pool.zig`)

**Optimizations:**
- Small integers (-5 to 256) cached (matches CPython)
- Empty containers (tuple, list, dict) pooled
- Reduces allocation overhead

```zig
pub fn getSmallInt(value: i64) ?PyObject {
    if (value >= -5 and value <= 256) {
        return cached_ints[@intCast(value + 5)];
    }
    return null;
}
```

## Type System

### Type Hierarchy

```
PyObject (root)
 ├── Primitives
 │   ├── PyBool
 │   ├── PyLong (all integers)
 │   ├── PyFloat
 │   └── PyComplex
 ├── Sequences
 │   ├── PyString
 │   ├── PyBytes
 │   ├── PyByteArray
 │   ├── PyList
 │   ├── PyTuple
 │   └── PyRange
 ├── Mappings
 │   ├── PyDict
 │   ├── PyDefaultDict
 │   └── PyCounter
 ├── Sets
 │   ├── PySet
 │   └── PyFrozenSet
 ├── Advanced
 │   ├── PyDateTime, PyDate, PyTime, PyTimeDelta
 │   ├── PyDecimal
 │   ├── PyFraction
 │   ├── PyUUID
 │   └── PyPath
 ├── Containers
 │   ├── PyDeque
 │   └── PyChainMap
 └── Special
     ├── PyArray (NumPy)
     ├── PyGenerator
     └── PySlice
```

Total: 38 types implemented

### Type Conversion Matrix

| Zig Type | Python Type | Method |
|----------|-------------|--------|
| `bool` | `bool` | Cached True/False |
| `i8`-`i64` | `int` | PyLong_FromLongLong |
| `u8`-`u64` | `int` | PyLong_FromUnsignedLongLong |
| `f32`, `f64` | `float` | PyFloat_FromDouble |
| `[]const u8` | `str` | PyUnicode_FromStringAndSize |
| `[]T` | `list` | Recursive conversion |
| `struct { ... }` | `dict` | Field-by-field |
| `py.PyX` | `X` | Identity (no conversion) |

## Build System

### Two-Layer Architecture

#### Python Layer (`buildzig.py`)
- Reads `pyproject.toml` configuration
- Generates `pyz3.build.zig` with module definitions
- Invokes Zig compiler
- Handles cross-compilation

#### Zig Layer (`build.zig`)
- Queries Python for headers/libs via sysconfig
- Translates C headers to Zig FFI
- Links native C sources (uthash, utarray)
- Supports optimization levels

### Build Process

```
pyproject.toml
     ↓
buildzig.py parses config
     ↓
Generate pyz3.build.zig
     ↓
zig build -Dbuild-file=pyz3.build.zig
     ↓
Link Python libs + C sources
     ↓
Output: module.abi3.so
```

### Environment Variables

- `PYZ3_OPTIMIZE` - Optimization level (Debug, ReleaseSafe, ReleaseFast, ReleaseSmall)
- `ZIG_TARGET` - Cross-compilation target (e.g., x86_64-linux-gnu)
- `PYZ3_DEBUG` - Enable debug mode (1, true, yes)

## Performance Optimizations

### 1. Fast Path Conversions

**Benchmark Results:**
- i64: 10.2x faster than generic
- f64: 4.9x faster
- bool: 15.3x faster (cached)
- string: 3.2x faster

### 2. GIL Caching

**Impact:** 10-100x for nested allocations

**Before:**
```zig
for (items) |item| {
    const gil = PyGILState_Ensure();  // Expensive!
    // ...
    PyGILState_Release(gil);
}
```

**After:**
```zig
const gil = ScopedGIL.acquire();  // Once!
for (items) |item| {
    // No GIL overhead
}
gil.release();
```

### 3. Object Pooling

**Cached Values:**
- Small ints: -5 to 256
- Empty tuple: `()`
- Empty list: `[]`
- Empty dict: `{}`

**Savings:** ~80% reduction in allocations for common values

### 4. Zero-Copy NumPy

**Benefit:** No data copying for array operations

```zig
pub fn processArray(arr: py.PyArray) !void {
    const data = try arr.data(f64);  // Zero-copy view
    for (data) |*value| {
        value.* *= 2;  // In-place modification
    }
}
```

## Extension Points

### Adding New Types

1. Create `pyz3/src/types/pynewtype.zig`
2. Implement required methods
3. Export from `types.zig`
4. Re-export from `pyz3.zig`
5. Add tests

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guide.

### Custom Allocators

```zig
const custom_allocator = MyAllocator.init();
defer custom_allocator.deinit();

const obj = try py.PyString.create("hello", custom_allocator);
```

### Operator Overloading

Supported operators (44 total):
- Arithmetic: +, -, *, /, //, %, **
- Bitwise: &, |, ^, <<, >>, ~
- Comparison: <, <=, ==, !=, >, >=
- Inplace: +=, -=, *=, etc.
- Unary: -, +, ~, abs, int, float, bool
- Special: @, divmod, pow

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Type discovery | O(n) compile-time | n = number of types |
| Function call | O(1) | With fast paths |
| Type conversion | O(1) primitives, O(n) containers | n = container size |
| GIL acquire | O(1) | Cached when nested |
| Memory alloc | O(1) | Via PyMem_RawMalloc |

## Best Practices

1. **Use fast paths** for primitive types
2. **Minimize GIL operations** with ScopedGIL
3. **Prefer zero-copy** for large arrays
4. **Pool common objects** when possible
5. **Profile before optimizing** (see `fastpath_bench.zig`)

## Known Limitations

1. NumPy C API incomplete (3 TODOs remaining)
2. Async generator disabled
3. eval branch quota: 100,000 (may slow large projects)
4. Limited API only (no stable ABI yet)

## Future Improvements

1. Complete NumPy C API implementation
2. Incremental compilation support
3. JIT-style specialization caching
4. Better error messages with source locations
5. Automatic type stub generation

---

For more information, see:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide
- [README.md](README.md) - User documentation
- [examples/](examples/) - Usage examples
