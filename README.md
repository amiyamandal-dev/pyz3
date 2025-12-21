# pyZ3 - High-Performance Python Extensions in Zig

<p align="center">
    <em>Build blazing-fast, memory-safe Python extensions in Zig with first-class NumPy integration and a seamless, modern developer experience.</em>
</p>
<p align="center">
    <em>Forked from the excellent <a href="https://github.com/fulcrum-so/ziggy-pydust">ziggy-pydust</a> project with a focus on data science and performance.</em>
</p>

<p align="center">
<a href="https://github.com/amiyamandal-dev/pyz3/actions/workflows/build-wheels.yml" target="_blank">
    <img src="https://img.shields.io/github/actions/workflow/status/amiyamandal-dev/pyz3/build-wheels.yml?branch=main&logo=github&label=build" alt="Build Status">
</a>
<a href="https://pypi.org/project/pyZ3" target="_blank">
    <img src="https://img.shields.io/pypi/v/pyZ3?color=blue" alt="PyPI Version">
</a>
<a href="https://pypi.org/project/pyZ3" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/pyz3" alt="Python Versions">
</a>
<a href="https://github.com/amiyamandal-dev/pyz3/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/pypi/l/pyZ3" alt="License">
</a>
</p>

---

**pyz3** combines Zig's performance and safety with Python's ease of use. It's designed for developers who need to speed up performance-critical code paths, from web services to data-intensive scientific computing, without the complexities of traditional C/C++ extensions.

## Key Features

- ⚡ **Blazing Fast**: Leverage Zig's performance with optimizations like GIL caching and direct FFI calls for 5-10x speedups.
- 🛡️ **Memory Safe**: Write safer code than C with Zig's compile-time checks and explicit memory management.
- 🤖 **Seamless `zigimport`**: Import `.zig` files directly in Python, just like `.py` files. No manual compilation needed for rapid development.
- 📊 **First-Class NumPy Integration**: Work with NumPy arrays with zero-copy data access and a type-safe API.
- 🛠️ **Modern CLI**: A powerful, Maturin-like CLI for creating, developing, and building your projects.
- 🔗 **C/C++ Interoperability**: Automatically manage and generate bindings for C/C++ dependencies.
- 📦 **Cross-Compilation**: Build and distribute wheels for Linux, macOS, and Windows with a single command.
- 🧪 **Integrated Testing**: Pytest plugin automatically discovers and runs your Zig tests.

## Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **Zig 0.15.x** ([Install Guide](https://ziglang.org/download/))

### 2. Installation

```bash
# Using uv (recommended)
uv pip install pyz3

# Or using pip
pip install pyz3
```

### 3. Create a New Project

The `pyz3 new` command scaffolds a complete, ready-to-use project.

```bash
pyz3 new my_fast_module
cd my_fast_module
```

This creates the following structure:
```
my_fast_module/
├── .gitignore
├── build.zig
├── my_fast_module/
│   └── __init__.py
├── pyproject.toml
├── README.md
├── src/
│   └── my_fast_module.zig  # Your Zig code lives here
└── tests/
    └── test_my_fast_module.py
```

### 4. Build and Test

Install in development mode and run the tests. `pyz3 develop` is like `pip install -e .` but also compiles your Zig code.

```bash
# Build the extension and install it in editable mode
pyz3 develop

# Run the pre-generated tests
pytest
```
You should see the tests pass! You've just compiled and tested your first Zig extension.

## Workflows

pyZ3 supports two powerful development workflows.

### Workflow 1: The `zigimport` Experience (for Rapid Development)

For the fastest iteration, `zigimport` lets you import `.zig` files directly, recompiling them automatically when they change. No terminals, no manual build commands.

**1. Create `main.py`:**
```python
# main.py
import pyz3.zigimport  # This enables the magic!
import my_fast_module

print(my_fast_module.add(10, 5))
```

**2. Run your Python script:**
```bash
python main.py
# Output: 15
```
Now, **edit `src/my_fast_module.zig` and save it**. Run `python main.py` again. `zigimport` detects the change, recompiles in the background, and your script runs the new code.

### Workflow 2: The CLI Experience (for Building & Distribution)

Use the CLI for building, testing, and packaging your extension.

- **Develop**: `pyz3 develop`
  - Builds your extension and installs it in an editable mode.
- **Watch**: `pyz3 watch --pytest`
  - Automatically rebuilds and re-runs tests when you save a file.
- **Build Wheels**: `pyz3 build-wheel --all-platforms`
  - Cross-compiles optimized wheels for Linux, macOS, and Windows, ready for distribution.

## NumPy Showcase

pyZ3's key strength is its deep integration with NumPy, providing zero-copy array access.

```zig
// src/my_fast_module.zig
const py = @import("pyz3");
const np = py.numpy;

// This function modifies the NumPy array in-place from Zig.
pub fn multiply_in_place(args: struct { 
    arr: np.PyArray(@This()),
    factor: f64,
}) !void {
    // Get a mutable slice of the NumPy array's data (zero-copy).
    const data = try args.arr.asSliceMut(f64);

    // Modify the data directly.
    for (data) |*val| {
        val.* *= args.factor;
    }
}
```

```python
# python_script.py
import numpy as np
import my_fast_module

# Create a NumPy array
my_array = np.array([1.0, 2.0, 3.0, 4.0])

# Pass it to Zig to be modified in-place
my_fast_module.multiply_in_place(my_array, 10.0)

# The original array is changed!
print(my_array)
# Output: [10. 20. 30. 40.]
```

## CLI Command Reference

| Command | Description |
|---|---|
| `pyz3 new <name>` | Create a new project from a template. |
| `pyz3 init` | Initialize pyz3 in an existing directory. |
| `pyz3 develop` | Build and install the package in editable mode. |
| `pyz3 watch` | Watch for file changes and rebuild automatically. |
| `pyz3 build-wheel` | Build distribution wheels, with cross-compilation. |
| `pyz3 add <url>` | Add a C/C++ dependency to your project. |
| `pyz3 list` | List all C/C++ dependencies. |
| `pyz3 build` | A lower-level command to build specific extensions. |

For more details, run `pyz3 --help` or `pyz3 <command> --help`.

## Contributing

Contributions are welcome! pyZ3 is an open-source project, and we appreciate any help, from documentation improvements to new features.

- **Development Guide**: See [DEVELOPMENT.md](DEVELOPMENT.md) for instructions on how to set up your development environment.
- **Roadmap**: Check out our [ROADMAP.md](docs/ROADMAP.md) to see what's next.
- **Bugs & Features**: Please open an issue on GitHub.

## Documentation

- **User Guide**: For tutorials and guides, see the [docs/guide](docs/guide) directory.
- **API Reference**: Check out the [docs/api](docs/api) for detailed API documentation.
- **Examples**: The [example/](example) directory contains many working examples.

## License

pyZ3 is licensed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.

## Acknowledgments

This project is a hard fork of **[ziggy-pydust](https://github.com/fulcrum-so/ziggy-pydust)**. We are deeply grateful to the original authors for creating an excellent foundation. For more details on the fork, see the [Fork Notice](docs/FORK_NOTICE.md).