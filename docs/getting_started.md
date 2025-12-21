# Getting Started with pyz3

pyz3 is a framework for building high-performance Python extensions in Zig. This guide will get you up and running in minutes.

## Prerequisites

- **Python 3.11+** (Python 3.12 and 3.13 supported)
- **Zig 0.15.x** ([Download](https://ziglang.org/download/))
- **Git** (for dependency management)

## Installation

### Option 1: Using uv (Recommended - Fast!)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install pyz3
uv pip install pyz3
```

### Option 2: Using pip

```bash
pip install pyz3
```

### Option 3: Using Poetry

```bash
poetry add pyz3
```

## Quick Start

### Create a New Project

The easiest way to get started is with the `pyz3 new` command:

```bash
# Create a new project
pyz3 new myproject

# Navigate to it
cd myproject

# Build and test
zig build
pytest
```

This creates a complete project structure:

```
myproject/
├── myproject.zig          # Your Zig code
├── pyproject.toml         # Project configuration
├── test/
│   └── test_myproject.py  # Tests
├── README.md
└── .gitignore
```

### Initialize in Existing Directory

If you already have a directory:

```bash
cd my-existing-project
pyz3 init -n mypackage --description "My awesome extension"
```

## Your First Extension

The generated `myproject.zig` contains a simple example:

```zig
const py = @import("pyz3");

pub fn hello(args: struct { name: []const u8 }) ![]const u8 {
    return "Hello, " ++ args.name ++ "!";
}

pub fn add(args: struct { a: i64, b: i64 }) i64 {
    return args.a + args.b;
}

comptime {
    py.rootmodule(@This());
}
```

## Building & Testing

### Option 1: Manual Build (Development)

```bash
# Build the extension
zig build

# Run tests
pytest
```

### Option 2: Auto-Import with zigimport

For rapid development, use zigimport to automatically compile on import:

```python
import pyz3.zigimport  # Enable auto-import
import myproject

# Your .zig file is automatically compiled!
print(myproject.hello("World"))  # "Hello, World!"
print(myproject.add(5, 3))        # 8
```

See [zigimport guide](guides/ZIGIMPORT_README.md) for advanced features like watch mode and dependency tracking.

### Option 3: Development Mode

Install your package in editable mode:

```bash
pyz3 develop
```

This builds the extension and installs it in development mode (like `pip install -e .`).

## Project Configuration

Your `pyproject.toml` is pre-configured:

```toml
[build-system]
requires = ["pyz3"]
build-backend = "pyz3.build"

[project]
name = "myproject"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pyz3"]

[tool.pyz3]
root = "."
build_zig = "build.zig"

[[tool.pyz3.ext_module]]
name = "myproject"
root = "myproject.zig"
```

## Next Steps

### Learn the Basics

1. **[Modules](guide/modules.md)** - Creating Python modules
2. **[Functions](guide/functions.md)** - Defining functions
3. **[Classes](guide/classes.md)** - Building classes
4. **[Exceptions](guide/exceptions.md)** - Error handling

### Advanced Features

- **[zigimport](guides/ZIGIMPORT_README.md)** - Auto-compilation and hot reload
- **[Memory Management](guide/_5_memory.md)** - Memory safety
- **[GIL Management](guide/gil.md)** - Concurrency
- **[Buffer Protocol](guide/_6_buffers.md)** - Zero-copy data exchange
- **[NumPy Integration](guide/numpy.md)** - Working with arrays

### Development Workflow

1. **Watch Mode** - Auto-rebuild on changes:
   ```bash
   pyz3 watch --test
   ```

2. **Type Checking** - Add type hints:
   ```bash
   mypy .
   ```

3. **Testing** - Run tests with coverage:
   ```bash
   pytest --cov=myproject --cov-report=html
   ```

4. **Building Wheels** - For distribution:
   ```bash
   pyz3 build-wheel --optimize=ReleaseFast
   ```

## IDE Setup

### VS Code

Recommended extensions:
- **Zig Language** - Zig syntax highlighting
- **Python** - Python support
- **Pylance** - Type checking

### PyCharm

1. Install Zig plugin
2. Configure Python interpreter to use your project's venv
3. Mark `test/` as test sources

## Troubleshooting

### "Zig not found"

Make sure Zig is in your PATH:

```bash
zig version  # Should print: 0.15.x
```

### "Module not found" after build

Ensure you're running Python from the project directory:

```bash
cd /path/to/myproject
python -c "import myproject"
```

Or install in development mode:

```bash
pyz3 develop
```

### Build errors

Check that your Zig code compiles:

```bash
zig build
```

Enable verbose output:

```bash
zig build --verbose
```

### Import errors with zigimport

Check the build directory:

```bash
ls ~/.zigimport/
```

Enable verbose mode:

```bash
export ZIGIMPORT_VERBOSE=1
python -c "import pyz3.zigimport; import myproject"
```

## Common Workflows

### Adding C Dependencies

```bash
pyz3 add https://github.com/user/c-library
```

### Publishing to PyPI

```bash
# Build wheels for all platforms
pyz3 build-wheel --all-platforms

# Check the wheels
pyz3 check

# Upload to PyPI
pyz3 deploy
```

### Running Examples

```bash
# Run example modules
cd example/
python -c "import pyz3.zigimport; import hello; print(hello.add(1, 2))"
```

## Getting Help

- **[Documentation Index](INDEX.md)** - Complete documentation
- **[Development Guide](../DEVELOPMENT.md)** - Contributing guide
- **[GitHub Issues](https://github.com/amiyamandal-dev/pyz3/issues)** - Report bugs
- **[Examples](../example/)** - Example code

## Version Information

**Current Version**: 0.8.0  
**Python Support**: 3.11, 3.12, 3.13  
**Zig Version**: 0.15.x  
**License**: Apache 2.0

---

**Ready to build something awesome?** Check out the [examples](../example/) directory for inspiration!
