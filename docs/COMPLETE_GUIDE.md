# PyZ3 Complete Guide: Development to Production

**A step-by-step guide for building Python extensions with Zig**

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Your First Extension](#your-first-extension)
5. [Project Structure](#project-structure)
6. [Writing Zig Code for Python](#writing-zig-code-for-python)
7. [Advanced Features](#advanced-features)
8. [NumPy Integration](#numpy-integration)
9. [Building and Testing](#building-and-testing)
10. [Packaging for Distribution](#packaging-for-distribution)
11. [Production Deployment](#production-deployment)
12. [Troubleshooting](#troubleshooting)

---

## Introduction

**What is PyZ3?**

PyZ3 lets you write Python extensions in Zig instead of C/C++. You get:
- **Speed**: Zig is as fast as C
- **Safety**: Better memory safety than C
- **Simplicity**: Cleaner syntax than C++
- **Modern**: Built-in tooling and package management

**When to use PyZ3?**

✅ Use PyZ3 when you need to:
- Speed up performance-critical Python code
- Work with low-level system operations
- Process large amounts of data efficiently
- Integrate with C libraries
- Build NumPy-compatible numerical operations

❌ Don't use PyZ3 if:
- Pure Python is fast enough
- You're just calling existing C libraries (use ctypes/cffi)
- Your bottleneck is I/O, not CPU

---

## Prerequisites

### Required Software

1. **Python 3.8+**
   ```bash
   python --version  # Should show 3.8 or higher
   ```

2. **Zig 0.15.2** (exact version required)
   ```bash
   # macOS
   brew install zig@0.15.2

   # Linux
   wget https://ziglang.org/download/0.15.2/zig-linux-x86_64-0.15.2.tar.xz
   tar -xf zig-linux-x86_64-0.15.2.tar.xz
   sudo mv zig-linux-x86_64-0.15.2 /opt/zig
   export PATH=/opt/zig:$PATH

   # Verify
   zig version  # Should show 0.15.2
   ```

3. **Poetry** (Python package manager)
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   poetry --version
   ```

### Optional but Recommended

- **Git**: For version control
- **A code editor**: VS Code, Vim, or any editor with Zig support
- **NumPy**: If you plan to work with numerical arrays

---

## Installation

### Step 1: Install PyZ3

```bash
# Option 1: From PyPI (when published)
pip install pyz3

# Option 2: From source (development)
git clone https://github.com/yourusername/pyz3.git
cd pyz3
pip install -e .
```

### Step 2: Verify Installation

```bash
python -c "import pyz3; print(pyz3.__version__)"
```

You should see the version number printed.

### Step 3: Set Up Your Development Environment

```bash
# Create a new directory for your project
mkdir my-fast-extension
cd my-fast-extension

# Initialize a Python project
poetry init --no-interaction
poetry add pyz3
```

---

## Your First Extension

Let's create a simple "Hello, World!" extension.

### Step 1: Create Project Structure

```bash
my-fast-extension/
├── pyproject.toml
├── src/
│   └── hello.zig
└── my_extension/
    └── __init__.py
```

### Step 2: Write the Configuration

**pyproject.toml:**
```toml
[build-system]
requires = ["pyz3>=0.8.0"]
build-backend = "pyz3.build"

[project]
name = "my-fast-extension"
version = "0.1.0"
description = "My first PyZ3 extension"

[tool.pyz3.ext-module.hello]
# This tells PyZ3 to build src/hello.zig into a Python module
root = "src/hello.zig"
```

### Step 3: Write Your First Zig Code

**src/hello.zig:**
```zig
const std = @import("std");
const py = @import("pyz3");

/// This function will be callable from Python as hello.greet(name)
pub fn greet(name: py.PyString) !py.PyString {
    // Get the name as a Zig string
    const name_str = try name.asSlice();

    // Create a greeting message
    const greeting = try std.fmt.allocPrint(
        py.allocator,
        "Hello, {s}! Welcome to PyZ3!",
        .{name_str}
    );
    defer py.allocator.free(greeting);

    // Return as a Python string
    return py.PyString.fromSlice(greeting);
}

/// This is required - it registers this file as a Python module
comptime {
    py.rootmodule(@This());
}
```

### Step 4: Create Python Wrapper

**my_extension/__init__.py:**
```python
"""My Fast Extension - A PyZ3 Example"""

# The compiled module will be available as 'hello'
from . import hello

__all__ = ['hello']
```

### Step 5: Build and Test

```bash
# Build the extension
python -m pyz3 build

# Test it interactively
python
```

```python
>>> from my_extension import hello
>>> result = hello.greet("World")
>>> print(result)
Hello, World! Welcome to PyZ3!
```

**Congratulations!** You just created your first PyZ3 extension! 🎉

---

## Project Structure

A typical PyZ3 project looks like this:

```
my-project/
├── pyproject.toml          # Project configuration
├── README.md               # Documentation
├── src/                    # Zig source files
│   ├── main.zig           # Main module
│   ├── utils.zig          # Helper functions
│   └── math_ops.zig       # Math operations
├── include/                # C header files (optional)
│   └── mylib.h
├── my_package/             # Python package
│   ├── __init__.py
│   └── helpers.py         # Pure Python helpers
├── test/                   # Tests
│   ├── test_basic.py
│   └── test_advanced.py
└── examples/               # Example usage
    └── demo.py
```

### Configuration File (pyproject.toml)

Here's a complete example:

```toml
[build-system]
requires = ["pyz3>=0.8.0"]
build-backend = "pyz3.build"

[project]
name = "my-awesome-extension"
version = "1.0.0"
description = "A high-performance Python extension"
authors = [{name = "Your Name", email = "you@example.com"}]
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20.0",  # If you use NumPy
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

# Define your extension modules
[tool.pyz3.ext-module.main]
root = "src/main.zig"

[tool.pyz3.ext-module.math_ops]
root = "src/math_ops.zig"
# Optional: Link against C libraries
libraries = ["m"]  # Math library

[tool.pyz3.ext-module.with_c_code]
root = "src/with_c.zig"
c_sources = ["src/helper.c"]
include_dirs = ["include"]
```

---

## Writing Zig Code for Python

### Basic Data Types

#### Strings

```zig
const py = @import("pyz3");

pub fn string_example(input: py.PyString) !py.PyString {
    // Convert Python string to Zig string
    const zig_str = try input.asSlice();

    // Do something with it
    const upper = try std.ascii.allocUpperString(py.allocator, zig_str);
    defer py.allocator.free(upper);

    // Return as Python string
    return py.PyString.fromSlice(upper);
}
```

#### Numbers

```zig
pub fn add_numbers(a: i64, b: i64) i64 {
    return a + b;
}

pub fn multiply_floats(a: f64, b: f64) f64 {
    return a * b;
}
```

#### Lists

```zig
pub fn process_list(items: py.PyList) !py.PyList {
    const result = try py.PyList.new();

    // Iterate through the list
    const len = try items.len();
    for (0..len) |i| {
        const item = try items.getItem(i);

        // Convert to i64, multiply by 2, append
        const num = try item.asLong();
        const doubled = num * 2;
        try result.append(try py.PyLong.from(doubled));
    }

    return result;
}
```

#### Dictionaries

```zig
pub fn create_dict() !py.PyDict {
    const dict = try py.PyDict.new();

    try dict.setItem(
        try py.PyString.fromSlice("name"),
        try py.PyString.fromSlice("PyZ3")
    );

    try dict.setItem(
        try py.PyString.fromSlice("version"),
        try py.PyLong.from(1)
    );

    return dict;
}
```

### Functions with Multiple Arguments

```zig
pub fn calculate(x: f64, y: f64, operation: py.PyString) !f64 {
    const op = try operation.asSlice();

    if (std.mem.eql(u8, op, "add")) {
        return x + y;
    } else if (std.mem.eql(u8, op, "multiply")) {
        return x * y;
    } else if (std.mem.eql(u8, op, "divide")) {
        if (y == 0) return error.DivisionByZero;
        return x / y;
    }

    return error.InvalidOperation;
}
```

### Optional Arguments and Keyword Arguments

```zig
pub fn greet_with_title(
    name: py.PyString,
    title: ?py.PyString,  // Optional argument
) !py.PyString {
    const name_str = try name.asSlice();

    if (title) |t| {
        const title_str = try t.asSlice();
        const msg = try std.fmt.allocPrint(
            py.allocator,
            "{s} {s}",
            .{title_str, name_str}
        );
        defer py.allocator.free(msg);
        return py.PyString.fromSlice(msg);
    } else {
        return name;
    }
}
```

### Error Handling

```zig
pub fn divide(a: f64, b: f64) !f64 {
    if (b == 0) {
        // This will become a Python ZeroDivisionError
        return py.PyError.raise(
            py.ZeroDivisionError,
            "Cannot divide by zero!"
        );
    }
    return a / b;
}

pub fn safe_operation(value: i64) !i64 {
    if (value < 0) {
        return py.PyError.raise(
            py.ValueError,
            "Value must be non-negative"
        );
    }

    if (value > 1000000) {
        return py.PyError.raise(
            py.ValueError,
            "Value too large (max: 1000000)"
        );
    }

    return value * value;
}
```

---

## Advanced Features

### Working with Classes

```zig
const std = @import("std");
const py = @import("pyz3");

/// Define a Python class in Zig
pub const Counter = struct {
    count: i64,

    /// Constructor - called when Counter() is invoked in Python
    pub fn __init__(initial: ?i64) !Counter {
        return Counter{
            .count = initial orelse 0,
        };
    }

    /// Instance method
    pub fn increment(self: *Counter) void {
        self.count += 1;
    }

    /// Instance method with parameter
    pub fn add(self: *Counter, value: i64) void {
        self.count += value;
    }

    /// Property getter
    pub fn getValue(self: *const Counter) i64 {
        return self.count;
    }

    /// String representation
    pub fn __repr__(self: *const Counter) !py.PyString {
        const str = try std.fmt.allocPrint(
            py.allocator,
            "Counter(count={})",
            .{self.count}
        );
        defer py.allocator.free(str);
        return py.PyString.fromSlice(str);
    }
};

comptime {
    py.rootmodule(@This());
}
```

Python usage:
```python
from my_extension import Counter

c = Counter(10)
c.increment()
c.add(5)
print(c.getValue())  # 16
print(c)  # Counter(count=16)
```

### Working with Buffers and Memory

```zig
pub fn process_bytes(data: py.PyBytes) !py.PyBytes {
    // Get raw bytes
    const bytes = try data.asSlice();

    // Allocate new buffer
    var result = try py.allocator.alloc(u8, bytes.len);
    defer py.allocator.free(result);

    // Process (example: XOR with 0x42)
    for (bytes, 0..) |byte, i| {
        result[i] = byte ^ 0x42;
    }

    // Return as Python bytes
    return py.PyBytes.fromSlice(result);
}

pub fn create_bytearray(size: usize) !py.PyByteArray {
    const arr = try py.PyByteArray.new(size);

    // Fill with pattern
    for (0..size) |i| {
        try arr.setItem(i, @intCast(i % 256));
    }

    return arr;
}
```

### Async/Await Support

```zig
pub fn async_operation(duration: f64) !py.PyCoroutine {
    // Create a coroutine that sleeps for duration seconds
    return py.PyCoroutine.create(struct {
        pub fn run(dur: f64) !py.PyObject {
            // This will be awaitable in Python
            const asyncio = try py.PyModule.import("asyncio");
            const sleep = try asyncio.getAttr("sleep");
            return try sleep.call(.{dur});
        }
    }.run, .{duration});
}
```

Python usage:
```python
import asyncio
from my_extension import async_operation

async def main():
    await async_operation(2.0)
    print("Done!")

asyncio.run(main())
```

### Working with Files and Paths

```zig
pub fn read_file_size(filepath: py.PyPath) !i64 {
    const path = try filepath.asSlice();

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    return @intCast(stat.size);
}

pub fn list_directory(dir_path: py.PyString) !py.PyList {
    const path = try dir_path.asSlice();
    const result = try py.PyList.new();

    var dir = try std.fs.cwd().openDir(path, .{ .iterate = true });
    defer dir.close();

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        try result.append(try py.PyString.fromSlice(entry.name));
    }

    return result;
}
```

---

## NumPy Integration

### Basic NumPy Array Operations

```zig
const std = @import("std");
const py = @import("pyz3");

/// Add a scalar to all elements in a NumPy array
pub fn add_scalar(array: py.PyObject, scalar: f64) !py.PyObject {
    const np = try py.PyModule.import("numpy");

    // Convert to numpy array if needed
    const np_array = try np.getAttr("asarray");
    const arr = try np_array.call(.{array});

    // Add scalar
    const add = try arr.getAttr("__add__");
    return try add.call(.{scalar});
}

/// Element-wise multiplication of two arrays
pub fn multiply_arrays(a: py.PyObject, b: py.PyObject) !py.PyObject {
    const np = try py.PyModule.import("numpy");

    const multiply = try np.getAttr("multiply");
    return try multiply.call(.{a, b});
}
```

### Creating NumPy Arrays from Zig

```zig
pub fn create_array(rows: usize, cols: usize) !py.PyObject {
    const np = try py.PyModule.import("numpy");

    // Create zeros array
    const zeros = try np.getAttr("zeros");
    return try zeros.call(.{.{rows, cols}});
}

pub fn create_range(start: i64, stop: i64, step: i64) !py.PyObject {
    const np = try py.PyModule.import("numpy");

    const arange = try np.getAttr("arange");
    return try arange.call(.{start, stop, step});
}
```

### High-Performance NumPy Processing

```zig
const std = @import("std");
const py = @import("pyz3");

/// Fast matrix multiplication using raw buffer access
pub fn fast_matrix_multiply(
    a_obj: py.PyObject,
    b_obj: py.PyObject
) !py.PyObject {
    // Get NumPy module
    const np = try py.PyModule.import("numpy");

    // Ensure arrays are contiguous and float64
    const ascontiguousarray = try np.getAttr("ascontiguousarray");
    const a = try ascontiguousarray.call(.{a_obj});
    const b = try ascontiguousarray.call(.{b_obj});

    // Get array properties
    const a_shape = try a.getAttr("shape");
    const b_shape = try b.getAttr("shape");

    const a_rows = try (try a_shape.getItem(0)).asLong();
    const a_cols = try (try a_shape.getItem(1)).asLong();
    const b_cols = try (try b_shape.getItem(1)).asLong();

    // Create result array
    const zeros = try np.getAttr("zeros");
    const result = try zeros.call(.{.{a_rows, b_cols}});

    // Get data pointers (unsafe but fast!)
    const a_data = try a.getAttr("ctypes");
    const b_data = try b.getAttr("ctypes");
    const r_data = try result.getAttr("ctypes");

    const a_ptr = try (try a_data.getAttr("data")).asLong();
    const b_ptr = try (try b_data.getAttr("data")).asLong();
    const r_ptr = try (try r_data.getAttr("data")).asLong();

    const a_array: [*]f64 = @ptrFromInt(@as(usize, @intCast(a_ptr)));
    const b_array: [*]f64 = @ptrFromInt(@as(usize, @intCast(b_ptr)));
    const r_array: [*]f64 = @ptrFromInt(@as(usize, @intCast(r_ptr)));

    // Perform matrix multiplication
    for (0..@intCast(a_rows)) |i| {
        for (0..@intCast(b_cols)) |j| {
            var sum: f64 = 0;
            for (0..@intCast(a_cols)) |k| {
                const a_idx = i * @as(usize, @intCast(a_cols)) + k;
                const b_idx = k * @as(usize, @intCast(b_cols)) + j;
                sum += a_array[a_idx] * b_array[b_idx];
            }
            const r_idx = i * @as(usize, @intCast(b_cols)) + j;
            r_array[r_idx] = sum;
        }
    }

    return result;
}

comptime {
    py.rootmodule(@This());
}
```

### SIMD-Accelerated Operations

```zig
const std = @import("std");
const py = @import("pyz3");

/// Sum array elements using SIMD
pub fn fast_sum(array: py.PyObject) !f64 {
    // Get data pointer and length
    const data_attr = try array.getAttr("ctypes");
    const data_ptr_long = try (try data_attr.getAttr("data")).asLong();
    const data_ptr: [*]f64 = @ptrFromInt(@as(usize, @intCast(data_ptr_long)));

    const size_obj = try array.getAttr("size");
    const size = try size_obj.asLong();
    const n: usize = @intCast(size);

    // SIMD vector of 4 f64s
    const Vec = @Vector(4, f64);
    var sum_vec = Vec{0, 0, 0, 0};

    // Process 4 elements at a time
    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        const vec: Vec = data_ptr[i..][0..4].*;
        sum_vec += vec;
    }

    // Sum the vector elements
    var total: f64 = 0;
    total += sum_vec[0];
    total += sum_vec[1];
    total += sum_vec[2];
    total += sum_vec[3];

    // Handle remaining elements
    while (i < n) : (i += 1) {
        total += data_ptr[i];
    }

    return total;
}

comptime {
    py.rootmodule(@This());
}
```

Python usage:
```python
import numpy as np
from my_extension import fast_sum, fast_matrix_multiply

# Test SIMD sum
arr = np.random.rand(1000000)
result = fast_sum(arr)
print(f"Sum: {result}")

# Test fast matrix multiplication
a = np.random.rand(1000, 1000)
b = np.random.rand(1000, 1000)
c = fast_matrix_multiply(a, b)
print(f"Result shape: {c.shape}")
```

---

## Building and Testing

### Development Build

```bash
# Build in debug mode (fast compilation, slower runtime)
python -m pyz3 build

# Build in release mode (slow compilation, fast runtime)
python -m pyz3 build --release

# Watch mode - rebuilds on file changes
python -m pyz3 watch
```

### Running Tests

Create **test/test_basic.py**:
```python
import pytest
from my_extension import greet, add_numbers

def test_greet():
    result = greet("Alice")
    assert "Alice" in result
    assert result.startswith("Hello")

def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0

def test_error_handling():
    from my_extension import divide

    with pytest.raises(ZeroDivisionError):
        divide(10, 0)
```

Run tests:
```bash
# Install pytest
pip install pytest pytest-cov

# Run tests
pytest test/

# With coverage
pytest test/ --cov=my_extension --cov-report=html
```

### Benchmarking

Create **benchmark/bench_performance.py**:
```python
import time
import numpy as np
from my_extension import fast_sum

# Pure Python version
def python_sum(arr):
    return sum(arr)

# Benchmark
sizes = [1000, 10000, 100000, 1000000]

for size in sizes:
    arr = np.random.rand(size)

    # Python version
    start = time.time()
    result_py = python_sum(arr)
    time_py = time.time() - start

    # Zig version
    start = time.time()
    result_zig = fast_sum(arr)
    time_zig = time.time() - start

    speedup = time_py / time_zig
    print(f"Size {size}: Python={time_py:.4f}s, Zig={time_zig:.4f}s, Speedup={speedup:.2f}x")
```

---

## Packaging for Distribution

### Step 1: Prepare for Release

Update **pyproject.toml**:
```toml
[project]
name = "my-awesome-extension"
version = "1.0.0"
description = "A high-performance Python extension"
readme = "README.md"
license = {text = "MIT"}
authors = [{name = "Your Name", email = "you@example.com"}]
keywords = ["performance", "numpy", "zig"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
```

### Step 2: Build Wheels

```bash
# Install build tools
pip install build twine

# Build source distribution and wheel
python -m build

# This creates:
# dist/my-awesome-extension-1.0.0.tar.gz
# dist/my_awesome_extension-1.0.0-cp311-cp311-macosx_11_0_arm64.whl
```

### Step 3: Test the Package

```bash
# Create a fresh virtual environment
python -m venv test_env
source test_env/bin/activate

# Install from the wheel
pip install dist/my_awesome_extension-1.0.0-*.whl

# Test it
python -c "from my_extension import greet; print(greet('World'))"
```

### Step 4: Publish to PyPI

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ my-awesome-extension

# If everything works, upload to real PyPI
twine upload dist/*
```

---

## Production Deployment

### Docker Deployment

Create **Dockerfile**:
```dockerfile
FROM python:3.11-slim

# Install Zig
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://ziglang.org/download/0.15.2/zig-linux-x86_64-0.15.2.tar.xz && \
    tar -xf zig-linux-x86_64-0.15.2.tar.xz && \
    mv zig-linux-x86_64-0.15.2 /opt/zig && \
    rm zig-linux-x86_64-0.15.2.tar.xz

ENV PATH="/opt/zig:${PATH}"

# Copy and install
WORKDIR /app
COPY . .
RUN pip install -e .

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t my-app .
docker run -p 8000:8000 my-app
```

### CI/CD with GitHub Actions

Create **.github/workflows/build.yml**:
```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install Zig
      uses: goto-bus-stop/setup-zig@v2
      with:
        version: 0.15.2

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]

    - name: Build extension
      run: python -m pyz3 build

    - name: Run tests
      run: pytest test/

    - name: Build wheel
      run: python -m build

    - name: Upload wheel
      uses: actions/upload-artifact@v3
      with:
        name: wheels
        path: dist/*.whl
```

### Performance Optimization

**Build with optimizations:**
```bash
# Maximum optimization
python -m pyz3 build --release

# With specific Zig optimization flags
ZIG_FLAGS="-O ReleaseFast" python -m pyz3 build
```

**Profile your code:**
```python
import cProfile
import pstats
from my_extension import expensive_function

# Profile
cProfile.run('expensive_function()', 'stats.prof')

# View results
stats = pstats.Stats('stats.prof')
stats.sort_stats('cumulative')
stats.print_stats(10)
```

---

## Troubleshooting

### Common Issues

#### Issue: "Zig compiler not found"

**Solution:**
```bash
# Verify Zig installation
which zig
zig version

# Add to PATH if needed
export PATH="/opt/zig:$PATH"
```

#### Issue: "ImportError: undefined symbol"

**Solution:**
```bash
# Rebuild in debug mode to see detailed errors
python -m pyz3 build --debug

# Check for ABI compatibility
python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
```

#### Issue: "Segmentation fault"

**Solution:**
1. Check for memory safety issues in Zig code
2. Use address sanitizer:
   ```bash
   ASAN_OPTIONS=detect_leaks=1 python your_script.py
   ```
3. Enable debug mode:
   ```zig
   const std = @import("std");
   pub fn problematic_function() !void {
       std.debug.print("Debug: entering function\n", .{});
       // Your code here
   }
   ```

#### Issue: "Build fails on different OS"

**Solution:**
- Use conditional compilation:
  ```zig
  const builtin = @import("builtin");

  pub fn platform_specific() !void {
      if (builtin.os.tag == .windows) {
          // Windows-specific code
      } else if (builtin.os.tag == .linux) {
          // Linux-specific code
      } else if (builtin.os.tag == .macos) {
          // macOS-specific code
      }
  }
  ```

### Performance Debugging

**Check if your extension is actually being used:**
```python
import my_extension
print(my_extension.__file__)  # Should show .so or .pyd file
```

**Measure actual speedup:**
```python
import timeit

python_time = timeit.timeit(
    'python_function(data)',
    setup='from my_module import python_function, data',
    number=1000
)

zig_time = timeit.timeit(
    'zig_function(data)',
    setup='from my_extension import zig_function; from my_module import data',
    number=1000
)

print(f"Speedup: {python_time / zig_time:.2f}x")
```

### Getting Help

1. **Check the examples:** Look at the `example/` directory in the PyZ3 repository
2. **Read the API docs:** Run `python -c "import pyz3; help(pyz3)"`
3. **Search issues:** https://github.com/your-org/pyz3/issues
4. **Ask questions:** Create a GitHub discussion

---

## Next Steps

Now that you understand the basics:

1. **Read the API Reference** - Learn all available PyZ3 types and functions
2. **Study the examples** - Look at the `example/` directory for more complex cases
3. **Build something real** - Start with a small performance bottleneck in your code
4. **Share your work** - Publish your extension to PyPI
5. **Contribute** - Help improve PyZ3 by submitting PRs

Happy coding! 🚀
