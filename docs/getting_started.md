# Getting Started with pyz3

pyz3 is a framework for building high-performance Python extension modules in Zig. This guide will help you create your first extension.

## Prerequisites

- **Python**: 3.11, 3.12, or 3.13
- **Zig**: 0.15.x (automatically installed with pyz3)
- **pip/uv**: For package management

## Installation

```bash
# Using pip
pip install pyz3

# Using uv (recommended)
uv pip install pyz3
```

## Quick Start

### 1. Create a New Project

```bash
# Create a new directory and initialize
mkdir my-extension && cd my-extension
pyz3 init -n my_extension --no-interactive

# Or with full options
pyz3 init \
  -n my_extension \
  -a "Your Name" \
  --email "you@example.com" \
  --description "My awesome Zig extension"
```

This creates a project structure:

```
my-extension/
├── src/
│   └── my_extension.zig    # Your Zig code
├── my_extension/
│   ├── __init__.py         # Python package
│   └── _lib.pyi            # Type stubs
├── test/
│   └── test_my_extension.py
├── pyproject.toml
├── build.py
└── README.md
```

### 2. Write Your First Function

Edit `src/my_extension.zig`:

```zig
const py = @import("pyz3");

/// A simple function that adds two numbers
pub fn add(args: struct { a: i64, b: i64 }) i64 {
    return args.a + args.b;
}

/// Fibonacci sequence
pub fn fibonacci(args: struct { n: u64 }) u64 {
    if (args.n < 2) return args.n;
    var a: u64 = 0;
    var b: u64 = 1;
    for (1..args.n) |_| {
        const tmp = a + b;
        a = b;
        b = tmp;
    }
    return b;
}

comptime {
    py.rootmodule(@This());
}
```

### 3. Build and Test

```bash
# Install in development mode
pip install -e .

# Or using uv
uv pip install -e .

# Run tests
pytest
```

### 4. Use in Python

```python
from my_extension import _lib

print(_lib.add(a=5, b=3))      # Output: 8
print(_lib.fibonacci(n=10))    # Output: 55
```

## Project Configuration

Your `pyproject.toml` defines the extension modules:

```toml
[build-system]
requires = ["poetry-core", "pyz3>=0.9.1"]
build-backend = "poetry.core.masonry.api"

[[tool.pyz3.ext_module]]
name = "my_extension._lib"      # Python import path
root = "src/my_extension.zig"   # Zig source file
limited_api = true              # Use stable ABI (recommended)
```

### Multiple Modules

```toml
[[tool.pyz3.ext_module]]
name = "mypackage.core"
root = "src/core.zig"

[[tool.pyz3.ext_module]]
name = "mypackage.utils"
root = "src/utils.zig"
```

## Type Mappings

pyz3 automatically converts between Python and Zig types:

| Zig Type | Python Type | Notes |
|----------|-------------|-------|
| `void` | `None` | |
| `bool` | `bool` | |
| `i8` - `i64` | `int` | Checked overflow |
| `u8` - `u64` | `int` | Checked overflow |
| `f32`, `f64` | `float` | |
| `[]const u8` | `str` | UTF-8 encoded |
| `?T` | `T \| None` | Optional types |
| `struct {...}` | Keyword args | For function parameters |
| `py.PyObject` | `object` | Any Python object |
| `py.PyList` | `list` | |
| `py.PyDict` | `dict` | |
| `py.PyTuple` | `tuple` | |
| `py.PyArray` | `numpy.ndarray` | Zero-copy access |

## Classes

Define Python classes with Zig structs:

```zig
const py = @import("pyz3");
const root = @This();

pub const Point = py.class(struct {
    pub const __doc__ = "A 2D point";
    const Self = @This();

    x: f64,
    y: f64,

    pub fn __init__(self: *Self, args: struct { x: f64, y: f64 }) void {
        self.x = args.x;
        self.y = args.y;
    }

    pub fn distance(self: *const Self) f64 {
        return @sqrt(self.x * self.x + self.y * self.y);
    }

    pub fn move(self: *Self, args: struct { dx: f64, dy: f64 }) void {
        self.x += args.dx;
        self.y += args.dy;
    }
});

comptime {
    py.rootmodule(root);
}
```

Usage in Python:

```python
from my_extension import _lib

p = _lib.Point(x=3.0, y=4.0)
print(p.distance())  # 5.0
p.move(dx=1.0, dy=0.0)
print(p.x)  # 4.0
```

## Exception Handling

Raise Python exceptions from Zig:

```zig
const py = @import("pyz3");
const root = @This();

pub fn divide(args: struct { a: i64, b: i64 }) !i64 {
    if (args.b == 0) {
        return py.ZeroDivisionError(root).raise("division by zero");
    }
    return @divTrunc(args.a, args.b);
}

pub fn validate_age(args: struct { age: i64 }) !void {
    if (args.age < 0) {
        return py.ValueError(root).raise("age cannot be negative");
    }
    if (args.age > 150) {
        return py.ValueError(root).raise("age seems unrealistic");
    }
}
```

Available exception types:
- `py.ValueError`
- `py.TypeError`
- `py.RuntimeError`
- `py.ZeroDivisionError`
- `py.IndexError`
- `py.KeyError`
- `py.AttributeError`
- `py.ImportError`
- `py.OSError`
- `py.MemoryError`

## NumPy Integration

Zero-copy access to NumPy arrays:

```zig
const py = @import("pyz3");
const root = @This();

pub fn sum_array(args: struct { arr: py.PyArray(root) }) !f64 {
    const data = try args.arr.asSlice(f64);
    var sum: f64 = 0;
    for (data) |val| {
        sum += val;
    }
    return sum;
}

pub fn double_inplace(args: struct { arr: py.PyArray(root) }) !void {
    const data = try args.arr.asSliceMut(f64);
    for (data) |*val| {
        val.* *= 2.0;
    }
}

pub fn create_range(args: struct { n: usize }) !py.PyArray(root) {
    var arr = try py.PyArray(root).create(.{ args.n }, f64);
    const data = try arr.asSliceMut(f64);
    for (data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    return arr;
}
```

## Testing

### Zig Tests

Add tests directly in your Zig file:

```zig
const std = @import("std");
const py = @import("pyz3");

pub fn add(args: struct { a: i64, b: i64 }) i64 {
    return args.a + args.b;
}

test "add works" {
    py.initialize();
    defer py.finalize();

    try std.testing.expectEqual(@as(i64, 5), add(.{ .a = 2, .b = 3 }));
}
```

Run Zig tests:

```bash
zig build test
```

### Python Tests

```python
# test/test_my_extension.py
from my_extension import _lib

def test_add():
    assert _lib.add(a=2, b=3) == 5

def test_fibonacci():
    assert _lib.fibonacci(n=10) == 55
```

Run all tests (Zig + Python):

```bash
pytest
```

## CLI Reference

```bash
pyz3 init [OPTIONS]      # Create new project
pyz3 build [OPTIONS]     # Build extension
pyz3 develop             # Install in development mode
pyz3 watch               # Watch mode with auto-rebuild
pyz3 test                # Run tests
pyz3 clean               # Clean build artifacts
pyz3 build-wheel         # Build distribution wheel
pyz3 deploy              # Publish to PyPI
```

## Next Steps

- [Classes Guide](guide/classes.md) - Class features including @classmethod and @staticmethod
- [NumPy Guide](guide/numpy.md) - NumPy integration details
- [Memory-Mapped Files](guide/mmap.md) - Zero-copy I/O and shared memory
- [New Features](guide/new_features.md) - Memory leak detection, watch mode, async/await
- [Memory Management](guide/_5_memory.md) - Memory safety
- [GIL Management](guide/gil.md) - Thread safety
- [Debugging](guide/debugging.md) - Debug your extensions
