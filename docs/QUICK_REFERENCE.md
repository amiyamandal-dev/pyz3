# pyz3 - Quick Reference

Fast reference for common tasks and commands.

## Installation

```bash
# Install pydust
pip install ziggy-pydust

# With distribution extras
pip install ziggy-pydust[dist]

# Install cookiecutter (required for project creation)
pip install cookiecutter
```

## Project Creation

```bash
# Interactive creation
pydust init

# Non-interactive
pydust init -n myproject --description "My extension" --email "me@example.com" --no-interactive

# Create in new directory
pydust new myproject
```

## Development Workflow

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# Or use pydust
pydust develop

# Watch mode (auto-rebuild)
pydust watch

# Watch with tests
pydust watch --test
pydust watch --pytest
```

## Building & Testing

```bash
# Run tests
pytest

# Build for current platform
pydust build-wheel

# Build for all platforms
pydust build-wheel --all-platforms

# Build optimized
pydust build-wheel --optimize ReleaseFast
```

## Distribution

```bash
# Validate packages
pydust check
pydust check --strict

# Deploy to PyPI
pydust deploy --username __token__ --password $PYPI_TOKEN

# Deploy to custom repository
pydust deploy --repository https://custom.pypi.org --username user --password pass
```

## C/C++ Dependencies

```bash
# Add dependency from GitHub
pydust add https://github.com/user/repo

# With custom name
pydust add https://github.com/user/repo -n mylib

# Specify headers
pydust add https://github.com/user/repo --headers header1.h header2.h

# List dependencies
pydust list

# Remove dependency
pydust remove libname
```

## Common Code Patterns

### Basic Function

```zig
const py = @import("pydust");

pub fn greet(args: struct { name: []const u8 }) []const u8 {
    return "Hello, " ++ args.name;
}

comptime {
    py.rootmodule(@This());
}
```

### Function with Optional Args

```zig
pub fn processData(args: struct {
    data: []const i64,
    threshold: f64 = 0.5,        // Optional with default
    normalize: ?bool = null,      // Optional without default
}) ![]const f64 {
    // ...
}
```

### Class Definition

```zig
pub const Counter = struct {
    count: i64,

    pub fn __init__(args: struct { initial: i64 = 0 }) Counter {
        return .{ .count = args.initial };
    }

    pub fn increment(self: *Counter, args: struct { by: i64 = 1 }) void {
        self.count += args.by;
    }

    pub fn getValue(self: *const Counter) i64 {
        return self.count;
    }
};
```

### Error Handling

```zig
pub fn divide(args: struct { a: f64, b: f64 }) !f64 {
    if (args.b == 0) return error.DivisionByZero;
    return args.a / args.b;
}

// Python side:
// try:
//     result = module.divide(10, 0)
// except RuntimeError as e:
//     print(e)  # "DivisionByZero"
```

### Working with Python Types

```zig
const py = @import("pydust");

pub fn sumList(args: struct { values: py.PyList }) !f64 {
    var sum: f64 = 0;
    var iter = try args.values.iter();
    while (try iter.next(f64)) |val| {
        sum += val;
    }
    return sum;
}

pub fn createDict() !py.PyDict {
    var dict = try py.PyDict.new();
    try dict.setItem("key", 42);
    try dict.setItem("name", "value");
    return dict;
}
```

## Type Conversions

| Zig Type | Python Type | Notes |
|----------|-------------|-------|
| `i64`, `u64`, `i32`, etc. | `int` | Auto converted |
| `f64`, `f32` | `float` | Auto converted |
| `bool` | `bool` | Auto converted |
| `[]const u8` | `str` | UTF-8 string |
| `[]const T` | `list` | Array/slice to list |
| `?T` | `Optional[T]` | None or value |
| `!T` | May raise exception | Error handling |
| `struct {}` | `dict` | Named struct fields |
| `tuple {}` | `tuple` | Anonymous struct |

## CLI Help

```bash
# General help
pydust --help

# Command-specific help
pydust init --help
pydust build-wheel --help
pydust deploy --help
pydust add --help
```

## Project Structure

```
myproject/
├── .github/workflows/    # CI/CD
├── src/                  # Zig source
│   └── myproject.zig
├── myproject/            # Python package
│   ├── __init__.py
│   └── _lib.pyi         # Type stubs
├── test/                 # Tests
├── pyproject.toml        # Config
└── build.py             # Build script
```

## Troubleshooting

### Build Errors

```bash
# Clean build cache
rm -rf .zig-cache zig-out

# Rebuild
zig build

# Verbose output
pydust develop --verbose
```

### Import Errors

```bash
# Ensure installed in development mode
pip install -e .

# Check extension built
ls myproject/*.so  # Linux/macOS
ls myproject/*.pyd  # Windows
```

### Testing Issues

```bash
# Run specific test file
pytest test/test_mymodule.py

# Verbose output
pytest -v

# Show print statements
pytest -s
```

## Performance Tips

1. **Use Release Builds**: `--optimize ReleaseFast` for production
2. **Minimize Type Conversions**: Work with Zig types internally
3. **Batch Operations**: Process arrays/lists in bulk
4. **Release GIL**: For CPU-intensive code without Python calls
5. **Profile**: Use `py.gil()` and `py.nogil()` strategically

## Common Patterns

### Release GIL for CPU Work

```zig
pub fn heavyComputation(args: struct { data: []const f64 }) ![]f64 {
    // Release GIL during CPU-intensive work
    const nogil = py.nogil();
    defer nogil.acquire();

    // CPU-intensive computation here
    var result = try allocator.alloc(f64, args.data.len);
    for (args.data, 0..) |val, i| {
        result[i] = complexMath(val);
    }

    return result;
}
```

### Custom Exceptions

```zig
pub fn validate(args: struct { value: i64 }) !void {
    if (args.value < 0) {
        return py.ValueError(root).raise("Value must be positive");
    }
    if (args.value > 100) {
        return py.RuntimeError(root).raiseFmt("Value {d} exceeds maximum", .{args.value});
    }
}
```

## Documentation Links

- **Main Docs**: https://pydust.fulcrum.so/latest
- **API Reference**: https://pydust.fulcrum.so/latest/zig
- **GitHub**: https://github.com/fulcrum-so/ziggy-pydust
- **Examples**: See `example/` directory in repo

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share projects
- **Documentation**: Comprehensive guides and API reference

---

**Quick Commands Cheat Sheet**

```bash
pydust init                    # Create project (interactive)
pydust new <name>             # Create in new directory
pydust develop                # Build and install
pydust watch --test           # Auto-rebuild and test
pydust build-wheel            # Build distribution
pydust check --strict         # Validate package
pydust deploy                 # Upload to PyPI
```
