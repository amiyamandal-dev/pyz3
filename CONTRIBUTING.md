# Contributing to pyz3

Thank you for your interest in contributing to pyz3! This guide will help you get started.

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (3.13 recommended)
- **Zig 0.15.2**
- **Poetry** (optional but recommended)

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/amiyamandal-dev/pyz3.git
cd pyz3

# Install dependencies
poetry install

# Or using pip
pip install -e .

# Run diagnostics to verify setup
poetry run python -m pyz3.diagnostics
```

## 📁 Project Structure

```
pyz3/
├── pyz3/                    # Python package
│   ├── src/                 # Zig source code
│   │   ├── pyz3.zig        # Main entry point
│   │   ├── discovery.zig   # Compile-time introspection
│   │   ├── pytypes.zig     # Type system
│   │   ├── functions.zig   # Function wrapping
│   │   ├── trampoline.zig  # Type conversions
│   │   └── types/          # 38 Python type wrappers
│   ├── __main__.py         # CLI entry point
│   ├── buildzig.py         # Build orchestration
│   └── diagnostics.py      # Diagnostic tools
├── example/                 # Example modules (24)
├── test/                    # Test suite (22+ files)
├── build.zig               # Zig build configuration
├── pyproject.toml          # Python project configuration
└── run_all_tests.sh        # Comprehensive test runner
```

## 🔨 Development Workflow

### 1. Build the Project

```bash
# Build all example modules
zig build

# Build with optimization
PYZ3_OPTIMIZE=ReleaseFast zig build

# Build specific module
zig build -Dtest-debug-root=example/hello.zig
```

### 2. Run Tests

```bash
# Run all tests
./run_all_tests.sh

# Run specific test file
poetry run pytest test/test_hello.py -v

# Run with coverage
poetry run pytest test/ --cov=pyz3
```

### 3. Code Quality

```bash
# Format Zig code
zig fmt pyz3/src/

# Lint Python code
poetry run ruff check pyz3/

# Auto-fix issues
poetry run ruff check --fix pyz3/
```

## 📝 Adding a New Python Type

### Step 1: Create Type Wrapper

Create `pyz3/src/types/pynewtype.zig`:

```zig
const py = @import("../pyz3.zig");
const PyObject = py.PyObject;

/// Wrapper for Python NewType object.
pub const PyNewType = extern struct {
    obj: PyObject,

    const Self = @This();

    /// Create a new NewType instance
    pub fn create(value: i64) !PyNewType {
        const obj = ffi.PyNewType_FromLong(value) orelse return error.PyRaiseError;
        return .{ .obj = .{ .py = obj } };
    }

    /// Get value from NewType
    pub fn getValue(self: Self) !i64 {
        return ffi.PyNewType_AsLong(self.obj.py);
    }

    pub usingnamespace py.PyObjectMixin("newtype", "NewType");
};
```

### Step 2: Export from types.zig

Add to `pyz3/src/types.zig`:

```zig
pub const pynewtype = @import("types/pynewtype.zig");
pub const PyNewType = pynewtype.PyNewType;
```

### Step 3: Re-export from pyz3.zig

Add to `pyz3/src/pyz3.zig`:

```zig
pub const PyNewType = types.PyNewType;
```

### Step 4: Add Tests

Create `test/test_newtype.py`:

```python
def test_newtype_creation(example):
    """Test NewType can be created and used"""
    import example.newtype_example
    result = example.newtype_example.create_newtype(42)
    assert result == 42
```

### Step 5: Add Example

Create `example/newtype_example.zig`:

```zig
const py = @import("pyz3");

pub fn create_newtype(value: i64) !py.PyNewType {
    return try py.PyNewType.create(value);
}

comptime {
    py.rootmodule(@This());
}
```

### Step 6: Register in pyproject.toml

```toml
[[tool.pyz3.ext_module]]
name = "example.newtype_example"
root = "example/newtype_example.zig"
```

## 🐛 Debugging

### Enable Debug Mode

```bash
export PYZ3_DEBUG=1
poetry run pytest test/test_your_feature.py -v -s
```

### Use GDB/LLDB

```bash
# Build with debug symbols
PYZ3_OPTIMIZE=Debug zig build

# Run under debugger
lldb -- python -m pytest test/test_your_feature.py
```

### Common Issues

#### Issue: "evaluation exceeded 10000 backwards branches"

**Solution:** Increase eval branch quota in affected file:

```zig
comptime {
    @setEvalBranchQuota(50000);
}
```

#### Issue: "pointless discard of function parameter"

**Solution:** Remove `_ = parameter;` if the parameter is actually used later.

#### Issue: Module not found in tests

**Solution:** Ensure module is built and fixture imports it:

```python
# test/conftest.py
@pytest.fixture
def example():
    import example
    import example.your_module  # Add this
    return example
```

#### Issue: Fatal Python abort during interpreter shutdown with ArenaAllocator

**Problem:** Using `std.heap.ArenaAllocator` initialized with `py.allocator` in `__del__` methods causes fatal abort during Python interpreter shutdown.

**Root Cause:** During interpreter finalization, `py.allocator` may become invalid before Python object `__del__` methods are called, causing crashes when `arena.deinit()` tries to free memory.

**Solution:** Use direct allocations instead of ArenaAllocator:

```zig
// ❌ BAD: ArenaAllocator causes shutdown crashes
pub const BadExample = py.class(struct {
    arena: std.heap.ArenaAllocator,
    buffer: []u8,

    pub fn __init__(self: *Self) !void {
        self.arena = std.heap.ArenaAllocator.init(py.allocator);
        self.buffer = try self.arena.allocator().alloc(u8, 100);
    }

    pub fn __del__(self: *Self) void {
        self.arena.deinit();  // CRASH during interpreter shutdown!
    }
});

// ✅ GOOD: Direct allocations are safe
pub const GoodExample = py.class(struct {
    allocator: std.mem.Allocator,
    buffer: ?[]u8,
    is_finalized: bool,

    pub fn __init__(self: *Self) !void {
        self.allocator = py.allocator;
        self.buffer = try self.allocator.alloc(u8, 100);
        self.is_finalized = false;
    }

    pub fn __del__(self: *Self) void {
        if (!self.is_finalized) {
            if (self.buffer) |buf| {
                self.allocator.free(buf);
                self.buffer = null;
            }
            self.is_finalized = true;
        }
    }
});
```

**Key Principles:**
- Make `__del__` idempotent (safe to call multiple times)
- Use optional fields (`?[]u8`) to track freed resources
- Add `is_finalized` flag to prevent double-free
- Prefer direct allocations over arena allocators in Python-exposed classes

## 📚 Coding Conventions

### Zig Code

- **Naming:** Use `camelCase` for functions, `PascalCase` for types
- **Error Handling:** Use `!` return types, not `catch unreachable`
- **Memory:** Always use provided allocators, never `std.heap.page_allocator`
- **Comments:** Document public APIs with `///` doc comments

```zig
/// Converts a Python list to Zig slice.
///
/// Allocates memory using the provided allocator.
/// Caller owns returned memory.
pub fn listToSlice(
    comptime T: type,
    list: PyList,
    allocator: std.mem.Allocator
) ![]T {
    // Implementation
}
```

### Python Code

- **Formatting:** Use `ruff` for formatting and linting
- **Type Hints:** Always add type hints to public functions
- **Docstrings:** Use Google-style docstrings

```python
def build_module(name: str, optimize: str = "ReleaseSafe") -> Path:
    """Build a pyz3 extension module.

    Args:
        name: Module name (e.g., "example.hello")
        optimize: Optimization level (Debug, ReleaseSafe, ReleaseFast)

    Returns:
        Path to compiled .so file

    Raises:
        BuildError: If compilation fails
    """
    # Implementation
```

## 🧪 Testing Guidelines

### Unit Tests

- Test each function/class in isolation
- Use descriptive test names: `test_<feature>_<scenario>`
- Include both success and failure cases

### Integration Tests

- Test real-world usage patterns
- Verify memory management (no leaks)
- Check GIL handling in multi-threaded scenarios

### Performance Tests

- Benchmark critical paths (see `example/fastpath_bench.zig`)
- Compare against pure Python baseline
- Document expected speedups

## 📊 Performance Optimization

### Use Fast Paths

For primitive types, use direct FFI calls:

```zig
// FAST: Direct PyLong_FromLongLong
pub fn wrapI64(value: i64) PyError!PyObject {
    const obj = ffi.PyLong_FromLongLong(value) orelse return error.PyRaiseError;
    return .{ .py = obj };
}

// SLOW: Generic trampoline
// Avoid for hot paths
```

### Cache PyObject Instances

```zig
// Use object pool for common values
const small_ints = py.object_pool.getSmallInt(value);
```

### Minimize GIL Operations

```zig
// Good: Single GIL acquire for batch operation
const gil = py.mem.ScopedGIL.acquire();
defer gil.release();
for (items) |item| {
    try processItem(item);
}

// Bad: Multiple GIL acquires
for (items) |item| {
    const gil = py.mem.ScopedGIL.acquire();
    defer gil.release();
    try processItem(item);
}
```

## 🔍 Code Review Checklist

Before submitting a PR:

- [ ] All tests pass (`./run_all_tests.sh`)
- [ ] Code is formatted (`zig fmt`, `ruff format`)
- [ ] No new compiler warnings
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Performance impact considered
- [ ] Memory leaks checked (use leak detection tests)

## 📜 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## 🤝 Getting Help

- **Issues:** https://github.com/amiyamandal-dev/pyz3/issues
- **Discussions:** https://github.com/amiyamandal-dev/pyz3/discussions
- **Documentation:** https://pyz3.readthedocs.io

## 🌟 Recognition

Contributors will be added to the project's README and CONTRIBUTORS file.

Thank you for contributing to pyz3!
