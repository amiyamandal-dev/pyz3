# pyZ3 CLI Usage Examples

Quick reference guide for the new Maturin-like CLI commands.

## Installation

```bash
pip install pyZ3
```

## Command Comparison: Before and After

### Creating a New Project

**Before:**
```bash
# Manual setup
mkdir my_extension
cd my_extension
touch pyproject.toml build.zig README.md
mkdir src tests my_extension
# ... manually create all files ...
git init
```

**After (New CLI):**
```bash
pyz3 new my_extension
# Done! Everything created automatically
```

### Setting Up for Development

**Before:**
```bash
pip install -e .
# Build manually or use pytest
```

**After (New CLI):**
```bash
pyz3 develop
# Builds Zig extensions + installs in editable mode
```

### Building Wheels

**Before:**
```bash
python -m pyz3.wheel --all-platforms
```

**After (New CLI):**
```bash
pyz3 build-wheel --all-platforms
# Shorter and more intuitive!
```

## Complete Workflow Examples

### Example 1: Brand New Project

```bash
# Step 1: Create project
pyz3 new awesome_extension
cd awesome_extension

# Step 2: Develop
pyz3 develop

# Step 3: Test
pytest

# Step 4: Make changes to src/awesome_extension.zig
# ... edit your code ...

# Step 5: Rebuild
pyz3 develop

# Step 6: Build wheels for distribution
pyz3 build-wheel --all-platforms

# Step 7: Publish to PyPI
pip install twine
twine upload dist/*
```

### Example 2: Add pyZ3 to Existing Project

```bash
# Step 1: Navigate to your project
cd my_existing_project

# Step 2: Initialize pyZ3
pyz3 init

# Step 3: Create your Zig extension in src/
# ... write your Zig code ...

# Step 4: Configure in pyproject.toml
# Add [[tool.pyz3.ext_module]] entries

# Step 5: Develop
pyz3 develop

# Step 6: Test
pytest
```

### Example 3: Fast Iteration with Watch Mode

```bash
# Terminal 1: Start watch mode
pyz3 watch --pytest

# Terminal 2: Edit your code
vim src/my_module.zig

# Watch mode automatically rebuilds and runs tests!
```

### Example 4: Cross-Platform Distribution

```bash
# Build for multiple platforms from a single machine
pyz3 build-wheel --platform linux-x86_64
pyz3 build-wheel --platform macos-arm64
pyz3 build-wheel --platform windows-x64

# Or build all at once
pyz3 build-wheel --all-platforms

# Result: dist/ contains wheels for all platforms
ls dist/
# my_extension-0.1.0-cp311-cp311-linux_x86_64.whl
# my_extension-0.1.0-cp311-cp311-macosx_11_0_arm64.whl
# my_extension-0.1.0-cp311-cp311-win_amd64.whl
```

### Example 5: Optimized Builds

```bash
# Development (fast compile, includes debug info)
pyz3 develop --optimize Debug

# Release testing (optimized, with safety checks)
pyz3 develop --optimize ReleaseSafe

# Production (maximum performance)
pyz3 build-wheel --optimize ReleaseFast

# Size-optimized (for AWS Lambda, edge functions)
pyz3 build-wheel --optimize ReleaseSmall
```

## Command Quick Reference

### Project Creation

```bash
# Create new project
pyz3 new <name>              # Creates directory with name
pyz3 new <name> --path ~/dev # Create in specific location

# Initialize existing directory
pyz3 init                    # Use directory name
pyz3 init --name mylib       # Override package name
pyz3 init --force            # Overwrite existing files
```

### Development

```bash
# Build and install
pyz3 develop                      # Debug build
pyz3 develop -o ReleaseFast       # Release build
pyz3 develop --verbose            # Show detailed output
pyz3 develop --extras dev test    # Install with extras
pyz3 develop --build-only         # Just build, don't install
```

### Distribution

```bash
# Build wheels
pyz3 build-wheel                          # Current platform
pyz3 build-wheel --platform linux-x86_64  # Specific platform
pyz3 build-wheel --all-platforms          # All platforms
pyz3 build-wheel --optimize ReleaseSmall  # Optimize for size
pyz3 build-wheel --output-dir wheelhouse  # Custom output
pyz3 build-wheel --verbose                # Detailed output
```

### Other Commands

```bash
# Watch mode
pyz3 watch                   # Watch and rebuild
pyz3 watch --test           # Watch and run Zig tests
pyz3 watch --pytest         # Watch and run pytest
pyz3 watch -o ReleaseFast   # Watch with optimization

# Low-level build
pyz3 build src/module.zig   # Build specific module
pyz3 debug src/module.zig   # Compile with debug symbols
```

## Integration with CI/CD

### GitHub Actions

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install pyZ3 pytest

      - name: Build and test
        run: |
          pyz3 develop
          pytest

  build-wheels:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Build wheels
        run: |
          pip install pyZ3
          pyz3 build-wheel --all-platforms

      - uses: actions/upload-artifact@v3
        with:
          name: wheels
          path: dist/*.whl
```

### GitLab CI

```yaml
stages:
  - test
  - build

test:
  stage: test
  script:
    - pip install pyZ3 pytest
    - pyz3 develop
    - pytest

build:
  stage: build
  script:
    - pip install pyZ3
    - pyz3 build-wheel --all-platforms
  artifacts:
    paths:
      - dist/
```

## Tips and Tricks

### 1. Use Aliases

```bash
# Add to your ~/.bashrc or ~/.zshrc
alias pd='pyz3'
alias pdd='pyz3 develop'
alias pdw='pyz3 build-wheel'

# Then use:
pd new my_ext
pdd
pdw --all-platforms
```

### 2. Development Workflow

```bash
# Keep two terminals open:
# Terminal 1: Watch mode
pyz3 watch --pytest

# Terminal 2: Edit and commit
vim src/module.zig
git commit -am "Add new feature"
```

### 3. Quick Testing

```bash
# Build and test in one line
pyz3 develop && pytest -v
```

### 4. Check What Gets Created

```bash
# Before creating a project, see what would be created
pyz3 new test_project
ls -la test_project/
cat test_project/pyproject.toml
rm -rf test_project  # Clean up after exploring
```

### 5. Build for Specific Python Versions

Currently builds for the Python version you're using:

```bash
# Use different Python versions
python3.11 -m pyz3 develop
python3.12 -m pyz3 develop
```

## Troubleshooting

### "Command not found: pyz3"

```bash
# Make sure pyz3 is installed
pip install pyZ3

# Or use as module
python -m pyz3 --help
```

### "pyproject.toml not found"

```bash
# Make sure you're in a pyz3 project directory
pyz3 init  # Initialize current directory
```

### Build Fails

```bash
# Try verbose mode to see what's happening
pyz3 develop --verbose

# Check Zig installation
zig version

# Make sure dependencies are installed
pip install -e .
```

### Import Errors After Installation

```bash
# Verify installation
pip show your-package-name

# Reinstall
pip uninstall your-package-name
pyz3 develop
```

## See Also

- [CLI Reference](docs/CLI.md) - Detailed command documentation
- [Distribution Guide](docs/distribution.md) - Full distribution guide
- [Quick Start](docs/DISTRIBUTION_QUICKSTART.md) - Fast-track commands
- [Main Documentation](https://pyz3.fulcrum.so) - Complete pyZ3 docs

## Comparison with Other Tools

| Tool | Language | Create Project | Dev Install | Build Wheels |
|------|----------|----------------|-------------|--------------|
| **pyZ3** | Zig | `pyz3 new` | `pyz3 develop` | `pyz3 build-wheel` |
| Maturin | Rust | `maturin new` | `maturin develop` | `maturin build` |
| Cython | Python/C | Manual | `pip install -e .` | `python setup.py bdist_wheel` |
| pybind11 | C++ | Manual | `pip install -e .` | `python setup.py bdist_wheel` |

pyZ3 brings Zig's simplicity with Maturin's developer experience!
