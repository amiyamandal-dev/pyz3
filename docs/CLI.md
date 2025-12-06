# pyZ3 CLI Reference

pyZ3 includes a command-line interface (CLI) similar to [Maturin](https://github.com/PyO3/maturin) for easy project management and development.

## Installation

```bash
pip install pyZ3
```

After installation, the `pyz3` command will be available in your terminal.

## Commands Overview

| Command | Description |
|---------|-------------|
| `pyz3 new` | Create a new project with boilerplate |
| `pyz3 init` | Initialize pyz3 in an existing directory |
| `pyz3 develop` | Build and install in development mode |
| `pyz3 build-wheel` | Build distribution wheels |
| `pyz3 add` | Add a C/C++ library dependency |
| `pyz3 list` | List all C/C++ dependencies |
| `pyz3 remove` | Remove a C/C++ dependency |
| `pyz3 build` | Build extension modules directly |
| `pyz3 watch` | Watch for changes and rebuild automatically |
| `pyz3 debug` | Compile Zig file with debug symbols |

---

## `pyz3 new`

Create a new pyZ3 project in a new directory with all necessary boilerplate.

### Usage

```bash
pyz3 new <project-name> [OPTIONS]
```

### Arguments

- `<project-name>` - Name of the project (required)

### Options

- `-p, --path <PATH>` - Parent directory (defaults to current directory)

### Example

```bash
# Create a new project called "my_extension"
pyz3 new my_extension

# Create in a specific directory
pyz3 new my_extension --path ~/projects
```

### What Gets Created

```
my_extension/
├── .git/                    # Git repository
├── .gitignore               # Python/Zig gitignore
├── build.zig                # Zig build file
├── pyproject.toml           # Python project config
├── README.md                # Project documentation
├── my_extension/            # Python package
│   └── __init__.py
├── src/                     # Zig source files
│   └── my_extension.zig
└── tests/                   # Python tests
    └── test_my_extension.py
```

---

## `pyz3 init`

Initialize pyZ3 in an existing directory. Useful when you want to add pyZ3 to an existing Python project.

### Usage

```bash
pyz3 init [OPTIONS]
```

### Options

- `-n, --name <NAME>` - Package name (defaults to directory name)
- `-a, --author <AUTHOR>` - Author name (defaults to git config)
- `-f, --force` - Overwrite existing files

### Example

```bash
# Initialize in current directory
cd my_project
pyz3 init

# Initialize with custom name
pyz3 init --name my_custom_name

# Force overwrite existing files
pyz3 init --force
```

---

## `pyz3 develop`

Build Zig extension modules and install the package in development mode (editable install). This is similar to `pip install -e .` but also builds the Zig extensions first.

### Usage

```bash
pyz3 develop [OPTIONS]
```

### Options

- `-o, --optimize <LEVEL>` - Optimization level: `Debug`, `ReleaseSafe`, `ReleaseFast`, `ReleaseSmall` (default: `Debug`)
- `-v, --verbose` - Enable verbose output
- `-e, --extras <EXTRAS>` - Install optional extras (e.g., `dev`, `test`)
- `--build-only` - Only build extensions without installing

### Examples

```bash
# Basic development install
pyz3 develop

# Release build for performance testing
pyz3 develop --optimize ReleaseFast

# Install with dev extras
pyz3 develop --extras dev test

# Just build, don't install
pyz3 develop --build-only

# Verbose output for debugging
pyz3 develop --verbose
```

### What It Does

1. Builds Zig extension modules with specified optimization level
2. Installs the package in editable mode (`pip install -e .`)
3. Verifies the installation

This is the recommended command for day-to-day development!

---

## `pyz3 build-wheel`

Build distribution wheels for one or more platforms. This is an alias for `python -m pyz3.wheel` with a more convenient interface.

### Usage

```bash
pyz3 build-wheel [OPTIONS]
```

### Options

- `--platform <PLATFORM>` - Target platform (see table below)
- `--all-platforms` - Build for all supported platforms
- `--optimize <LEVEL>` - Optimization level (default: `ReleaseFast`)
- `--output-dir <DIR>` - Output directory (default: `dist`)
- `--no-clean` - Don't clean build artifacts before building
- `-v, --verbose` - Enable verbose output

### Supported Platforms

| Platform | Flag |
|----------|------|
| Linux x86_64 | `linux-x86_64` |
| Linux ARM64 | `linux-aarch64` |
| macOS x86_64 | `macos-x86_64` |
| macOS ARM64 | `macos-arm64` |
| Windows x64 | `windows-x64` |

### Examples

```bash
# Build for current platform
pyz3 build-wheel

# Build for specific platform
pyz3 build-wheel --platform linux-x86_64

# Build for all platforms
pyz3 build-wheel --all-platforms

# Optimized for size (good for AWS Lambda)
pyz3 build-wheel --optimize ReleaseSmall

# Custom output directory
pyz3 build-wheel --output-dir wheelhouse
```

### Cross-Compilation

pyZ3 uses Zig's built-in cross-compilation to build wheels for any platform from any platform:

```bash
# Build Linux wheels from macOS
pyz3 build-wheel --platform linux-x86_64

# Build macOS ARM64 wheels from Windows
pyz3 build-wheel --platform macos-arm64
```

---

## `pyz3 add`

Add a C/C++ library dependency to your project. Automatically clones the library, generates Zig bindings, and integrates it into your build system.

### Usage

```bash
pyz3 add <source> [OPTIONS]
```

### Arguments

- `<source>` - GitHub URL, Git URL, or local filesystem path

### Options

- `-n, --name <NAME>` - Override dependency name (defaults to repo/directory name)
- `--headers <HEADERS...>` - Specify main headers to expose (auto-detected if not provided)
- `-v, --verbose` - Enable verbose output

### Examples

```bash
# Add from GitHub
pyz3 add https://github.com/d99kris/rapidcsv

# Add with custom name
pyz3 add https://github.com/nlohmann/json --name json

# Add local library
pyz3 add /usr/local/include/sqlite3.h --name sqlite

# Specify headers
pyz3 add https://github.com/nothings/stb \
  --headers stb_image.h stb_image_write.h
```

### What It Does

1. **Clones** the library to `deps/<name>/`
2. **Discovers** headers and source files
3. **Generates** Zig bindings in `bindings/<name>.zig`
4. **Creates** build configuration in `bindings/deps.zig.inc`
5. **Generates** Python wrapper template in `src/<name>_wrapper.zig`
6. **Tracks** dependency metadata in `pyz3_deps.json`

### Generated Files

```
your-project/
├── deps/
│   └── rapidcsv/           # Cloned library
├── bindings/
│   ├── rapidcsv.zig        # Auto-generated Zig bindings
│   └── deps.zig.inc        # Build configuration
├── src/
│   └── rapidcsv_wrapper.zig # Python wrapper template
└── pyz3_deps.json        # Dependency tracking
```

---

## `pyz3 list`

List all C/C++ dependencies in the project.

### Usage

```bash
pyz3 list
```

### Example Output

```
Dependencies (2):

  • rapidcsv
    Version: v8.90
    Source: https://github.com/d99kris/rapidcsv
    Type: Header-only
    Headers: rapidcsv.h

  • json
    Version: v3.11.3
    Source: https://github.com/nlohmann/json
    Type: Header-only
    Headers: json.hpp
```

---

## `pyz3 remove`

Remove a C/C++ dependency from the project.

### Usage

```bash
pyz3 remove <name>
```

### Arguments

- `<name>` - Name of the dependency to remove

### Example

```bash
# Remove rapidcsv
pyz3 remove rapidcsv
```

This removes the dependency from tracking and deletes generated bindings. The cloned source in `deps/` and wrapper template are preserved for safety.

---

## `pyz3 build`

Lower-level command to build specific Zig extension modules. Most users should use `pyz3 develop` instead.

### Usage

```bash
pyz3 build [OPTIONS] <EXTENSIONS>
```

### Arguments

- `<EXTENSIONS>` - Space-separated list of extensions in format `<name>=<path>` or `<path>`

### Options

- `-z, --zig-exe <PATH>` - Custom Zig executable path
- `-b, --build-zig <FILE>` - Custom build.zig file (default: `build.zig`)
- `-m, --self-managed` - Use self-managed build mode
- `-a, --limited-api` - Use Python limited API (default: true)
- `-p, --prefix <PREFIX>` - Module name prefix

### Examples

```bash
# Build a single module
pyz3 build src/my_module.zig

# Build with custom name
pyz3 build mypackage.core=src/core.zig

# Build multiple modules
pyz3 build src/module1.zig src/module2.zig
```

---

## `pyz3 watch`

Watch Zig source files for changes and automatically rebuild. Great for iterative development.

### Usage

```bash
pyz3 watch [OPTIONS] [PYTEST_ARGS]
```

### Options

- `-o, --optimize <LEVEL>` - Optimization level (default: `Debug`)
- `-t, --test` - Run Zig tests after rebuild
- `--pytest` - Run pytest instead of Zig tests
- `<PYTEST_ARGS>` - Additional arguments for pytest (when using `--pytest`)

### Examples

```bash
# Watch and rebuild on changes
pyz3 watch

# Watch and run Zig tests
pyz3 watch --test

# Watch and run pytest
pyz3 watch --pytest

# Watch with specific pytest args
pyz3 watch --pytest -v tests/test_core.py

# Watch with release optimization
pyz3 watch --optimize ReleaseFast
```

---

## `pyz3 debug`

Compile a Zig file with debug symbols. Useful for IDE debugging.

### Usage

```bash
pyz3 debug <entrypoint>
```

### Arguments

- `<entrypoint>` - Zig file to compile with debug symbols

### Example

```bash
pyz3 debug src/my_module.zig
```

---

## Comparison with Maturin

If you're familiar with Maturin (for PyO3/Rust), here's how pyZ3 commands map:

| Maturin | pyZ3 | Notes |
|---------|--------|-------|
| `maturin new` | `pyz3 new` | Creates new project |
| `maturin init` | `pyz3 init` | Initializes existing directory |
| `maturin develop` | `pyz3 develop` | Development install |
| `maturin build` | `pyz3 build-wheel` | Builds wheels |
| `maturin publish` | *(use twine)* | Publish to PyPI |

---

## Typical Workflows

### Starting a New Project

```bash
# 1. Create project
pyz3 new my_extension
cd my_extension

# 2. Develop and test
pyz3 develop
pytest

# 3. Make changes and rebuild
# Edit src/my_extension.zig
pyz3 develop

# 4. Build release wheels
pyz3 build-wheel --all-platforms
```

### Adding pyZ3 to Existing Project

```bash
# 1. Initialize
cd existing_project
pyz3 init

# 2. Write your Zig code in src/
# Edit src/my_module.zig

# 3. Develop
pyz3 develop
```

### Development with Auto-Rebuild

```bash
# Terminal 1: Watch mode
pyz3 watch --pytest

# Terminal 2: Edit code
# Changes trigger automatic rebuild and test
```

### Building for Production

```bash
# Build optimized wheels for all platforms
pyz3 build-wheel --all-platforms --optimize ReleaseSmall

# Wheels in dist/
ls dist/*.whl
```

---

## Environment Variables

Some pyZ3 commands respect environment variables:

- `ZIG_TARGET` - Override target platform (e.g., `x86_64-linux-gnu`)
- `PYZ3_OPTIMIZE` - Override optimization level
- `PYTHON` - Python executable to use

---

## Getting Help

For any command, use `--help`:

```bash
pyz3 --help
pyz3 new --help
pyz3 develop --help
pyz3 build-wheel --help
```

---

## See Also

- [Distribution Guide](./distribution.md) - Detailed guide on building and distributing wheels
- [Quick Start Guide](./DISTRIBUTION_QUICKSTART.md) - Fast-track commands
- [Main Documentation](https://pyz3.fulcrum.so) - Full pyZ3 documentation
