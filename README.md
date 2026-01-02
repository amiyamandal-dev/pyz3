# pyz3 - Python Extensions in Zig

<p align="center">
    <em>A high-performance framework for writing Python extension modules in Zig with automatic memory management, hot-reload, and NumPy integration.</em>
</p>
<p align="center">
    <em>🌟 Inspired by <a href="https://github.com/fulcrum-so/ziggy-pydust">ziggy-pydust</a></em>
</p>

<p align="center">
<a href="https://github.com/amiyamandal-dev/pyz3/actions" target="_blank">
    <img src="https://img.shields.io/github/actions/workflow/status/yourusername/pyz3/ci.yml?branch=main&logo=github" alt="Actions">
</a>
<a href="https://pypi.org/project/pyz3" target="_blank">
    <img src="https://img.shields.io/pypi/v/pyz3" alt="Package version">
</a>
<a href="https://docs.python.org/3/whatsnew/3.11.html" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/pyz3" alt="Python version">
</a>
<a href="https://github.com/amiyamandal-dev/pyz3/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/yourusername/pyz3" alt="License">
</a>
</p>

---

**Documentation**: <a href="https://github.com/amiyamandal-dev/pyz3" target="_blank">https://github.com/amiyamandal-dev/pyz3</a>

**Source Code**: <a href="https://github.com/amiyamandal-dev/pyz3" target="_blank">https://github.com/amiyamandal-dev/pyz3</a>

---

## Overview

pyz3 is a complete framework for building high-performance Python extension modules in Zig. It provides:

- 🚀 **Seamless Python-Zig Interop** - Automatic argument marshalling and type conversion
- 📊 **NumPy Integration** - Call NumPy functions from Zig via Python object interface
- 🔧 **Complete CLI Toolkit** - Maturin-style commands for project lifecycle management
- 📦 **Cross-Platform Builds** - Build wheels for Linux, macOS, and Windows
- 🔗 **C/C++ Integration** - Automatic binding generation for C/C++ libraries
- 🧪 **Testing Integration** - Pytest plugin to discover and run Zig tests
- ⚡ **Hot Reload** - Watch mode with automatic rebuilding
- 🛡️ **Memory Safe** - Leverages Zig's safety features with Python's GC

## Quick Example

```zig
const py = @import("pyz3");

pub fn fibonacci(args: struct { n: u64 }) u64 {
    if (args.n < 2) return args.n;

    var sum: u64 = 0;
    var last: u64 = 0;
    var curr: u64 = 1;
    for (1..args.n) |_| {
        sum = last + curr;
        last = curr;
        curr = sum;
    }
    return sum;
}

comptime {
    py.rootmodule(@This());
}
```

```python
import mymodule
print(mymodule.fibonacci(10))  # Output: 55
```

## NumPy Integration Example

NumPy can be used from Zig code via the Python object interface:

```zig
const py = @import("pyz3");

const root = @This();

/// Create and manipulate NumPy arrays from Zig
pub fn array_stats() !py.PyObject {
    const np = try py.numpy.getModule(@This());
    defer np.decref();

    // Create arange(1, 11)
    const arange_method = try np.getAttribute("arange");
    defer arange_method.decref();
    const arr = try py.call(root, py.PyObject, arange_method, .{ 1, 11 }, .{});
    defer arr.decref();

    // Get mean
    const mean_method = try arr.getAttribute("mean");
    defer mean_method.decref();
    return try py.call(root, py.PyObject, mean_method, .{}, .{});
}

comptime {
    py.rootmodule(root);
}
```

```python
import mymodule

result = mymodule.array_stats()
print(result)  # Output: 5.5
```

> **Note:** Direct zero-copy array access via `PyArray` is not yet implemented. Arrays are currently accessed through Python method calls.

## Compatibility

- **Zig**: 0.15.x (tested with 0.15.1)
- **Python**: 3.11, 3.12, 3.13 (CPython)
- **Platforms**: Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x64)
- **pyz3 Version**: 0.9.3

## Installation

```bash
pip install pyz3
```

Or with distribution extras for building wheels:

```bash
pip install pyz3[dist]
```

## Quick Start

### 1. Create a New Project

```bash
# Create project using cookiecutter template
pyz3 init -n myproject --description "My awesome extension" --email "you@example.com" --no-interactive

cd myproject
```

### 2. Build Your Extension

```bash
# Development build
zig build

# Release build
zig build -Doptimize=ReleaseFast

# Watch mode (hot reload)
pyz3 watch
```

### 3. Test Your Extension

```bash
# Run pytest
pytest

# Run specific test
pytest test/test_myproject.py -v
```

### 4. Package for Distribution

```bash
# Build wheel for current platform
python -m build --wheel

# Build for all platforms (uses cross-compilation)
pyz3 build-wheel --all-platforms

# Publish to PyPI
pyz3 deploy --repository testpypi  # Test first!
pyz3 deploy --repository pypi       # Production
```

## CLI Commands

pyz3 provides a complete CLI for managing your extension projects:

```bash
pyz3 init [OPTIONS]           # Initialize new project
pyz3 build [OPTIONS]          # Build extension module
pyz3 watch                    # Watch mode with hot reload
pyz3 test [OPTIONS]           # Run tests
pyz3 clean                    # Clean build artifacts
pyz3 build-wheel [OPTIONS]        # Build distribution packages
pyz3 deploy [OPTIONS]        # Publish to PyPI
```

## Key Features

### Type-Safe Python-Zig Bridge

Automatic conversion between Python and Zig types:

| Zig Type | Python Type |
|----------|-------------|
| `void` | `None` |
| `bool` | `bool` |
| `i32`, `i64` | `int` |
| `f32`, `f64` | `float` |
| `[]const u8` | `str` |
| `struct {...}` | `dict` |
| `py.PyObject` | `numpy.ndarray` (via `py.numpy.getModule`) |

### Classes and Methods

```zig
pub const Point = py.class(struct {
    pub const __doc__ = "A 2D point";
    const Self = @This();

    x: f64,
    y: f64,

    pub fn __init__(self: *Self, args: struct { x: f64, y: f64 }) !void {
        self.* = .{ .x = args.x, .y = args.y };
    }

    pub fn distance(self: *const Self) f64 {
        return @sqrt(self.x * self.x + self.y * self.y);
    }
});
```

### Exception Handling

```zig
pub fn divide(args: struct { a: i64, b: i64 }) !i64 {
    if (args.b == 0) {
        return py.ZeroDivisionError(root).raise("division by zero");
    }
    return @divTrunc(args.a, args.b);
}
```

## Cross-Platform Distribution

Build wheels for multiple platforms:

```bash
# Using environment variables
ZIG_TARGET=x86_64-linux-gnu PYZ3_OPTIMIZE=ReleaseFast python -m build --wheel
ZIG_TARGET=aarch64-linux-gnu PYZ3_OPTIMIZE=ReleaseFast python -m build --wheel
ZIG_TARGET=x86_64-macos PYZ3_OPTIMIZE=ReleaseFast python -m build --wheel
ZIG_TARGET=aarch64-macos PYZ3_OPTIMIZE=ReleaseFast python -m build --wheel
ZIG_TARGET=x86_64-windows-gnu PYZ3_OPTIMIZE=ReleaseFast python -m build --wheel
```

The build system automatically:
- Detects target platform
- Cross-compiles for different architectures
- Creates manylinux-compatible wheels
- Handles platform-specific optimizations

## Performance

pyz3 leverages Zig's performance advantages:

- **Zero-cost abstractions** - No runtime overhead
- **Compile-time optimizations** - Zig's comptime for metaprogramming
- **SIMD support** - Automatic vectorization where possible
- **Small binaries** - Smaller than equivalent Rust extensions
- **Fast compilation** - Faster than Rust, comparable to C

## Acknowledgments

This project is a hard fork of [ziggy-pydust](https://github.com/fulcrum-so/ziggy-pydust) by [Fulcrum](https://fulcrum.so).

Major differences in pyz3:
- ✅ NumPy integration via Python object interface (zero-copy PyArray planned)
- ✅ Enhanced cross-compilation support
- ✅ Updated CLI commands and workflows
- ✅ NumPy examples and tests
- ✅ Improved documentation for data science use cases

Special thanks to the original ziggy-pydust contributors for creating an excellent foundation!

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- **Original Project**: [ziggy-pydust](https://github.com/fulcrum-so/ziggy-pydust)
- **Zig Language**: [ziglang.org](https://ziglang.org)
- **NumPy**: [numpy.org](https://numpy.org)
