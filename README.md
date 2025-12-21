# pyz3 - Python Extensions in Zig

<p align="center">
    <em>A high-performance framework for writing Python extension modules in Zig with automatic memory management, hot-reload, and NumPy integration.</em>
</p>
<p align="center">
    <em>🌟 Inspired by <a href="https://github.com/fulcrum-so/ziggy-pydust">ziggy-pydust</a></em>
</p>

<p align="center">
<a href="https://github.com/amiyamandal-dev/pyz3/actions" target="_blank">
    <img src="https://img.shields.io/github/actions/workflow/status/amiyamandal-dev/pyz3/ci.yml?branch=main&logo=github" alt="Actions">
</a>
<a href="https://pypi.org/project/pyz3" target="_blank">
    <img src="https://img.shields.io/pypi/v/pyz3" alt="Package version">
</a>
<a href="https://docs.python.org/3/whatsnew/3.11.html" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/pyz3" alt="Python version">
</a>
<a href="https://github.com/amiyamandal-dev/pyz3/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/amiyamandal-dev/pyz3" alt="License">
</a>
</p>

---

**Documentation**: <a href="https://github.com/amiyamandal-dev/pyz3" target="_blank">https://github.com/amiyamandal-dev/pyz3</a>

**Source Code**: <a href="https://github.com/amiyamandal-dev/pyz3" target="_blank">https://github.com/amiyamandal-dev/pyz3</a>

---

## Overview

pyz3 is a complete framework for building high-performance Python extension modules in Zig. It provides:

- 🚀 **Seamless Python-Zig Interop** - Automatic argument marshalling and type conversion
- 📊 **NumPy Integration** - Zero-copy array access with type-safe dtype mapping
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

```zig
const py = @import("pyz3");

pub fn double_array(args: struct { arr: py.PyArray(@This()) }) !py.PyArray(@This()) {
    // Zero-copy access to NumPy array
    const data = try args.arr.asSliceMut(f64);

    for (data) |*val| {
        val.* *= 2.0;
    }

    return args.arr;
}

comptime {
    py.rootmodule(@This());
}
```

```python
import numpy as np
import mymodule

arr = np.array([1.0, 2.0, 3.0])
result = mymodule.double_array(arr)
print(result)  # Output: [2.0, 4.0, 6.0]
```

## Compatibility

- **Zig**: 0.15.x (tested with 0.15.2)
- **Python**: 3.11+ (CPython)
- **Platforms**: Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x64)

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
# Create a new project
pyz3 init -n myproject --description "My awesome extension" --email "you@example.com"

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
| `py.PyArray(root)` | `numpy.ndarray` |

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
- ✅ Built-in NumPy integration with zero-copy array access
- ✅ Enhanced cross-compilation support
- ✅ Updated CLI commands and workflows
- ✅ Comprehensive NumPy examples and tests
- ✅ Improved documentation for data science use cases

Special thanks to the original ziggy-pydust contributors for creating an excellent foundation!

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

To set up a development environment for contributing to pyz3:

#### 1. Clone the Repository

```bash
git clone https://github.com/amiyamandal-dev/pyz3.git
cd pyz3
```

#### 2. Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows

# Install uv (fast Python package installer)
pip install uv

# Install dependencies
uv pip install -r requirements.txt
```

#### 3. Install Development Dependencies

```bash
# Install distribution dependencies (optional, for building wheels)
uv pip install -r requirements-dist.txt
```

#### 4. Build the Project

```bash
# Build with Zig
zig build

# Or use Make
make build
```

#### 5. Run Tests

```bash
# Run all tests
./run_all_tests.sh

# Or run specific test suites
make test          # Python tests only
make test-zig      # Zig tests only
make test-all      # All tests

# Or use pytest directly
pytest test/ -v
```

#### 6. Quick Verification

```bash
# Quick 5-second verification check
./run_all_tests.sh --quick
```

### Project Structure

```
pyz3/
├── pyz3/               # Main Python package
│   ├── src/            # Core Zig source files
│   └── tests/          # Unit tests
├── example/            # Example Zig modules
├── test/               # Integration tests
├── docs/               # Documentation
├── build.zig           # Root build configuration
├── pytest.build.zig    # Pytest integration build
├── pyz3.build.zig      # PyZ3 build API
├── pyproject.toml      # Poetry configuration
├── requirements.txt    # Python dependencies (uv pip)
└── run_all_tests.sh    # Comprehensive test runner
```

### Available Commands

```bash
# Version management
make version                # Show current version
make bump-patch            # Bump patch version
make bump-minor            # Bump minor version
make bump-major            # Bump major version

# Testing
make test                  # Run Python tests
make test-zig              # Run Zig tests
make test-all              # Run all tests

# Building
make build                 # Build package
make clean                 # Clean build artifacts
make install               # Install in development mode
```

### Notes for Contributors

- The project uses both **Poetry** and **uv pip** for dependency management
- Python path is automatically detected from `.venv/bin/python`
- All tests should pass before submitting a PR
- Generated files (`.pyi` stubs, `.abi3.so` extensions) are gitignored
- Use `zig build` for development builds, `zig build -Doptimize=ReleaseFast` for production

## Links

- **Original Project**: [ziggy-pydust](https://github.com/fulcrum-so/ziggy-pydust)
- **Zig Language**: [ziglang.org](https://ziglang.org)
- **NumPy**: [numpy.org](https://numpy.org)
