# PyZ3 Documentation Index

**Complete documentation for PyZ3 - Write Python extensions in Zig**

## 📚 Documentation Structure

### Getting Started
1. **[Complete Guide](COMPLETE_GUIDE.md)** ⭐ START HERE
   - Installation and setup
   - Your first extension
   - Development to production workflow
   - Comprehensive tutorials

2. **[Quick Reference](QUICK_REFERENCE.md)**
   - Fast lookup for common operations
   - Type conversion cheat sheet
   - Code templates
   - Build commands

### Examples

#### Basic Examples
- **[01 - Hello World](examples/01_hello_world.md)**
  - Simplest possible extension
  - Basic string handling
  - Module structure

- **[02 - NumPy Operations](examples/02_numpy_operations.md)**
  - Array manipulation
  - SIMD operations
  - Matrix multiplication
  - Performance benchmarks

#### Advanced Examples
- **[03 - Image Processing](examples/03_real_world_image_processing.md)**
  - Real-world application
  - Grayscale conversion
  - Filters and effects
  - Edge detection

### API Reference

- **[Test Suite - All Types](../test/test_all_types.py)**
  - Complete type coverage
  - Usage examples for every type
  - Integration tests

### Development Resources

- **[Getting Started Guide](getting_started.md)**
  - Quick start tutorial
  - Basic concepts

- **[Development Guide](DEVELOPMENT.md)**
  - Contributing guidelines
  - Development setup
  - Code organization

## 🎯 Quick Navigation

### By Task

**I want to...**

- **Create my first extension** → [Complete Guide - Your First Extension](COMPLETE_GUIDE.md#your-first-extension)
- **Work with NumPy** → [NumPy Operations Example](examples/02_numpy_operations.md)
- **Handle strings and numbers** → [Quick Reference - Type Conversion](QUICK_REFERENCE.md#type-conversion-cheat-sheet)
- **Create classes** → [Quick Reference - Classes](QUICK_REFERENCE.md#classes)
- **Deploy to production** → [Complete Guide - Production Deployment](COMPLETE_GUIDE.md#production-deployment)
- **Troubleshoot errors** → [Complete Guide - Troubleshooting](COMPLETE_GUIDE.md#troubleshooting)
- **Optimize performance** → [Complete Guide - NumPy Integration](COMPLETE_GUIDE.md#numpy-integration)
- **Package for PyPI** → [Complete Guide - Packaging](COMPLETE_GUIDE.md#packaging-for-distribution)

### By Experience Level

#### Beginner (New to PyZ3)
1. Read [Complete Guide - Introduction](COMPLETE_GUIDE.md#introduction)
2. Follow [Your First Extension](COMPLETE_GUIDE.md#your-first-extension)
3. Try [Hello World Example](examples/01_hello_world.md)
4. Keep [Quick Reference](QUICK_REFERENCE.md) handy

#### Intermediate (Built a few extensions)
1. Study [NumPy Operations](examples/02_numpy_operations.md)
2. Learn [Advanced Features](COMPLETE_GUIDE.md#advanced-features)
3. Read [Working with Classes](QUICK_REFERENCE.md#classes)
4. Explore [All Types Tests](../test/test_all_types.py)

#### Advanced (Ready for production)
1. Optimize with [SIMD Operations](examples/02_numpy_operations.md#simd-accelerated-operations)
2. Follow [Production Deployment](COMPLETE_GUIDE.md#production-deployment)
3. Set up [CI/CD](COMPLETE_GUIDE.md#cicd-with-github-actions)
4. Review [Performance Tips](QUICK_REFERENCE.md#performance-tips)

## 📖 Documentation Highlights

### Type Coverage

All 38 Python types are documented with examples:

**Basic Types:**
- PyBool, PyBytes, PyString, PyLong, PyFloat

**Collections:**
- PyList, PyDict, PyTuple, PySet, PyFrozenSet

**Numeric:**
- PyComplex, PyDecimal, PyFraction

**Date/Time:**
- PyDateTime, PyDate, PyTime, PyTimeDelta

**Advanced:**
- PyPath, PyUUID, PyRange, PyGenerator

**Collections (Special):**
- PyDefaultDict, PyCounter, PyDeque, PyEnum

**Internal:**
- PyBuffer, PyCode, PyFrame, PyIter, PyMemoryView, PyModule, PyType, PyObject

**Async:**
- PyCoroutine, PyAwaitable, PyGIL

### Code Examples

#### Simple Function
```zig
pub fn greet(name: py.PyString) !py.PyString {
    const name_str = try name.asSlice();
    return py.PyString.fromSlice(
        try std.fmt.allocPrint(py.allocator, "Hello, {s}!", .{name_str})
    );
}
```

#### NumPy SIMD Operation
```zig
pub fn fast_sum(arr: py.PyObject) !f64 {
    // Get data pointer
    const data: [*]f64 = /* ... */;
    const n: usize = /* ... */;

    // SIMD sum
    const Vec = @Vector(4, f64);
    var sum_vec = Vec{0, 0, 0, 0};

    var i: usize = 0;
    while (i + 4 <= n) : (i += 4) {
        sum_vec += data[i..][0..4].*;
    }

    return /* total */;
}
```

#### Class Definition
```zig
pub const Counter = struct {
    count: i64,

    pub fn __init__(initial: i64) Counter {
        return .{ .count = initial };
    }

    pub fn increment(self: *Counter) void {
        self.count += 1;
    }
};
```

## 🚀 Common Workflows

### Development Workflow
```bash
# 1. Create project
mkdir my-extension && cd my-extension

# 2. Set up structure
# Create pyproject.toml and src/main.zig

# 3. Build
python -m pyz3 build

# 4. Test
python -c "import main; print(main.hello('World'))"

# 5. Package
python -m build

# 6. Publish
twine upload dist/*
```

### Testing Workflow
```bash
# Run tests
pytest test/

# With coverage
pytest test/ --cov=my_extension

# Benchmark
python benchmark/bench.py
```

### CI/CD Workflow
```yaml
# .github/workflows/test.yml
- uses: actions/setup-python@v4
- uses: goto-bus-stop/setup-zig@v2
  with:
    version: 0.15.2
- run: python -m pyz3 build
- run: pytest test/
```

## 🔧 Tools and Utilities

### Build Commands
```bash
python -m pyz3 build          # Debug build
python -m pyz3 build --release  # Release build
python -m pyz3 watch          # Watch mode
python -m pyz3 clean          # Clean artifacts
```

### Debugging
```bash
# Debug build with symbols
ZIG_FLAGS="-g" python -m pyz3 build

# Run with ASAN
ASAN_OPTIONS=detect_leaks=1 python script.py

# Profile
python -m cProfile -o stats.prof script.py
```

## 📊 Performance Benchmarks

Typical speedups compared to pure Python:

| Operation | Speedup |
|-----------|---------|
| String processing | 5-10x |
| Numerical loops | 10-50x |
| Array operations (SIMD) | 10-100x |
| Image processing | 3-20x |
| Matrix multiplication | 2-5x |

*Benchmarks vary based on problem size and optimization level*

## 🆘 Getting Help

1. **Check documentation**: Start with [Complete Guide](COMPLETE_GUIDE.md)
2. **Look at examples**: See [examples/](examples/)
3. **Search tests**: Find usage in [test_all_types.py](../test/test_all_types.py)
4. **Troubleshoot**: See [Troubleshooting](COMPLETE_GUIDE.md#troubleshooting)
5. **Ask questions**: GitHub Issues or Discussions

## 📝 Contributing

Want to improve the documentation?

1. Fix typos → Submit PR
2. Add examples → Submit PR with working code
3. Report issues → Open GitHub issue
4. Suggest improvements → Start a discussion

## 🔗 External Resources

- **Zig Documentation**: https://ziglang.org/documentation/0.15.2/
- **Python C API**: https://docs.python.org/3/c-api/
- **NumPy C API**: https://numpy.org/doc/stable/reference/c-api/
- **PyZ3 Repository**: https://github.com/your-org/pyz3

## 📅 Documentation Version

- **Last Updated**: 2025-12-21
- **PyZ3 Version**: 0.8.0+
- **Zig Version**: 0.15.2
- **Python Version**: 3.8+

---

**Ready to get started?** → [Complete Guide](COMPLETE_GUIDE.md)

**Need quick answers?** → [Quick Reference](QUICK_REFERENCE.md)

**Want to see code?** → [Examples](examples/)
