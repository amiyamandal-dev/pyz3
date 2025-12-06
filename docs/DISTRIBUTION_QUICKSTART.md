# Distribution Quick Start

Fast track to building and distributing your pyZ3 extension.

## Prerequisites

```bash
pip install pyZ3
```

## 0. Quick Start (New CLI)

pyZ3 now includes a Maturin-style CLI for easy project management:

```bash
# Create a new project
pyz3 new my_extension
cd my_extension

# Build and install in development mode
pyz3 develop

# Build wheels
pyz3 build-wheel --all-platforms
```

For detailed CLI documentation, see [CLI.md](./CLI.md).

## 1. Build a Wheel (Current Platform)

```bash
# Using new CLI (recommended)
pyz3 build-wheel

# Using Python module (alternative)
python -m pyz3.wheel

# Using shell script (alternative)
./scripts/build-wheels.sh
```

Output: `dist/your_package-0.1.0-cp311-cp311-macosx_11_0_arm64.whl`

## 2. Build for Specific Platform

```bash
# Using new CLI (recommended)
pyz3 build-wheel --platform linux-x86_64
pyz3 build-wheel --platform macos-arm64
pyz3 build-wheel --platform windows-x64

# Using Python module (alternative)
python -m pyz3.wheel --platform linux-x86_64
python -m pyz3.wheel --platform macos-arm64
python -m pyz3.wheel --platform windows-x64
```

## 3. Build for All Platforms

```bash
# Using new CLI (recommended)
pyz3 build-wheel --all-platforms

# Using Python module (alternative)
python -m pyz3.wheel --all-platforms
```

This creates wheels for:
- Linux (x86_64, aarch64)
- macOS (x86_64, arm64)
- Windows (x64)

## 4. Test Your Wheel

```bash
# Install the wheel
pip install dist/*.whl

# Test it
python -c "import your_module; your_module.test_function()"
```

## 5. Publish to PyPI

### Option A: Using Twine (Manual)

```bash
# Install twine
pip install twine

# Test on TestPyPI first
twine upload --repository testpypi dist/*

# Publish to PyPI
twine upload dist/*
```

### Option B: Using GitHub Actions (Automated)

1. **Configure PyPI trusted publishing:**
   - Go to PyPI → Your Project → Publishing
   - Add trusted publisher:
     - Owner: `your-github-username`
     - Repository: `your-repo-name`
     - Workflow: `build-wheels.yml`
     - Environment: `pypi`

2. **Tag and push a release:**
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. **GitHub Actions will automatically:**
   - Build wheels for all platforms
   - Test each wheel
   - Publish to PyPI
   - Create GitHub release

## 6. Optimization Levels

Choose based on your needs:

```bash
# Using new CLI (recommended)
pyz3 build-wheel --optimize ReleaseSmall   # Smallest binary (good for Lambda/Edge)
pyz3 build-wheel --optimize ReleaseFast    # Fastest execution (default)
pyz3 build-wheel --optimize ReleaseSafe    # Balanced (safe + fast)
pyz3 build-wheel --optimize Debug          # Development/debugging

# Using Python module (alternative)
python -m pyz3.wheel --optimize ReleaseSmall
python -m pyz3.wheel --optimize ReleaseFast
python -m pyz3.wheel --optimize ReleaseSafe
python -m pyz3.wheel --optimize Debug
```

## 7. Advanced: CI/CD Integration

### GitHub Actions (Built-in)

Already configured! Just push a tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

### GitLab CI

```yaml
# .gitlab-ci.yml
build:
  script:
    - pip install build
    - python -m pyz3.wheel --all-platforms
  artifacts:
    paths:
      - dist/
```

### Local Build Script

```python
# build.py
from pyz3.wheel import WheelBuilder, BuildConfig, Platform

builder = WheelBuilder()
config = BuildConfig(target_platform=Platform.LINUX_X86_64)
wheel = builder.build(config)
print(f"Built: {wheel}")
```

## 8. Troubleshooting

### "Module not found" after installation

Check the wheel installed correctly:
```bash
pip show your-package
pip list | grep your-package
```

### Cross-compilation fails

Ensure Zig is installed:
```bash
zig version  # Should be 0.15.2+
```

### Import error on Linux

Repair the wheel with `auditwheel`:
```bash
pip install auditwheel
auditwheel repair dist/*.whl --plat manylinux_2_17_x86_64 -w dist/
```

## Need Help?

- **Full Guide**: See [distribution.md](./distribution.md)
- **GitHub Issues**: https://github.com/fulcrum-so/pyZ3/issues
- **Examples**: Check `example/` directory

## Quick Command Reference

### New CLI (Recommended)

```bash
# Create new project
pyz3 new my_extension

# Initialize existing directory
pyz3 init

# Development install
pyz3 develop

# Build current platform
pyz3 build-wheel

# Build all platforms
pyz3 build-wheel --all-platforms

# Build specific platform
pyz3 build-wheel --platform linux-x86_64

# Set optimization
pyz3 build-wheel --optimize ReleaseSmall

# Verbose output
pyz3 build-wheel -v

# Help
pyz3 --help
pyz3 build-wheel --help
```

### Python Module (Alternative)

```bash
# Build current platform
python -m pyz3.wheel

# Build all platforms
python -m pyz3.wheel --all-platforms

# Build specific platform
python -m pyz3.wheel --platform linux-x86_64

# Set optimization
python -m pyz3.wheel --optimize ReleaseSmall
```

## Platform Tags

| Platform | Flag |
|----------|------|
| Linux x86_64 | `linux-x86_64` |
| Linux aarch64 | `linux-aarch64` |
| macOS x86_64 | `macos-x86_64` |
| macOS arm64 | `macos-arm64` |
| Windows x64 | `windows-x64` |

## That's It!

You're ready to distribute your pyZ3 extension. 🎉

For more advanced topics, see the [full distribution guide](./distribution.md).
