# Distribution & Cross-Compilation Guide

This guide covers building and distributing pyZ3 extensions for multiple platforms.

## Quick Start

### Build a wheel for your current platform

```bash
python -m pyz3.wheel
```

### Build for a specific platform

```bash
python -m pyz3.wheel --platform linux-x86_64
```

### Build for all platforms

```bash
python -m pyz3.wheel --all-platforms
```

## Supported Platforms

| Platform | Target Triple | Wheel Tag |
|----------|---------------|-----------|
| Linux x86_64 | `x86_64-linux-gnu` | `manylinux_2_17_x86_64` |
| Linux aarch64 | `aarch64-linux-gnu` | `manylinux_2_17_aarch64` |
| macOS x86_64 | `x86_64-macos` | `macosx_10_9_x86_64` |
| macOS arm64 | `aarch64-macos` | `macosx_11_0_arm64` |
| Windows x64 | `x86_64-windows-gnu` | `win_amd64` |

## Cross-Compilation

pyZ3 supports cross-compilation using Zig's built-in cross-compilation capabilities.

### Using Environment Variables

```bash
# Set target platform
export ZIG_TARGET=x86_64-linux-gnu

# Set optimization level
export PYDUST_OPTIMIZE=ReleaseFast

# Build
python -m build --wheel
```

### Using the wheel builder

```python
from pathlib import Path
from pyz3.wheel import WheelBuilder, BuildConfig, Platform

builder = WheelBuilder()

# Build for Linux x86_64
config = BuildConfig(
    target_platform=Platform.LINUX_X86_64,
    optimize="ReleaseFast",
    output_dir=Path("dist"),
)

wheel_path = builder.build(config)
print(f"Built: {wheel_path}")
```

## Building with GitHub Actions

The project includes a GitHub Actions workflow that automatically builds wheels for all supported platforms.

### Workflow Features

- **Multi-platform builds**: Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x64)
- **Multiple Python versions**: 3.9, 3.10, 3.11, 3.12, 3.13
- **Wheel repair**: Automatically repairs wheels using `auditwheel` (Linux) and `delocate` (macOS)
- **Testing**: Tests each wheel before uploading
- **Automatic PyPI publishing**: Publishes to PyPI on tagged releases

### Triggering Builds

The workflow runs on:
- Push to `main` or `develop` branches
- Pull requests
- Git tags starting with `v` (e.g., `v0.2.0`)
- Manual dispatch

### Creating a Release

```bash
# Tag a release
git tag v0.2.0
git push origin v0.2.0

# GitHub Actions will automatically:
# 1. Build wheels for all platforms
# 2. Test each wheel
# 3. Publish to PyPI (if configured)
# 4. Create a GitHub release with wheel artifacts
```

## Publishing to PyPI

### Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org
2. **API Token**: Generate an API token from your PyPI account settings
3. **Configure GitHub Secrets**: Add your PyPI token to GitHub repository secrets as `PYPI_API_TOKEN`

### Using Trusted Publishing (Recommended)

GitHub Actions workflow uses PyPI's trusted publishing feature, which doesn't require storing tokens.

1. Go to PyPI project settings
2. Add a trusted publisher:
   - **Owner**: Your GitHub username/org
   - **Repository**: Your repository name
   - **Workflow**: `build-wheels.yml`
   - **Environment**: `pypi`

### Manual Publishing

```bash
# Install twine
pip install twine

# Build wheels
python -m pyz3.wheel --all-platforms

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ your-package

# Upload to PyPI
twine upload dist/*
```

## Optimization Levels

Choose the optimization level based on your needs:

| Level | Use Case | Binary Size | Performance |
|-------|----------|-------------|-------------|
| `Debug` | Development, debugging | Largest | Slowest |
| `ReleaseSafe` | Production with safety checks | Large | Fast |
| `ReleaseFast` | Maximum performance | Medium | Fastest |
| `ReleaseSmall` | Size-constrained environments | Smallest | Fast |

```bash
# Build with specific optimization
python -m pyz3.wheel --optimize ReleaseSmall
```

## Troubleshooting

### Linux: `auditwheel` errors

```bash
# Install auditwheel
pip install auditwheel

# Repair wheel
auditwheel repair dist/mywheel.whl --plat manylinux_2_17_x86_64 -w dist/
```

### macOS: Library dependencies

```bash
# Install delocate
pip install delocate

# Check dependencies
delocate-listdeps dist/mywheel.whl

# Repair wheel
delocate-wheel -w dist/ dist/mywheel.whl
```

### Windows: Missing DLLs

Ensure all required DLLs are included in the wheel:

```python
# In setup.py
from setuptools import setup

setup(
    # ...
    package_data={
        'mypackage': ['*.dll'],
    },
)
```

### Cross-compilation fails

1. **Check Zig version**: Ensure you have Zig 0.15.2 or later
2. **Verify target triple**: Use `zig targets` to see supported targets
3. **Check Python version**: Ensure Python development headers are available

## Advanced: Custom Build Configuration

Create a custom build script:

```python
# build_custom.py
from pyz3.wheel import WheelBuilder, BuildConfig, Platform

def build_custom_wheels():
    """Build wheels with custom configuration."""
    builder = WheelBuilder()

    platforms = [
        Platform.LINUX_X86_64,
        Platform.MACOS_ARM64,
    ]

    for platform in platforms:
        config = BuildConfig(
            target_platform=platform,
            optimize="ReleaseFast",
        )

        print(f"\nBuilding for {platform.value}...")
        wheel = builder.build(config, clean=True, verbose=True)
        print(f"✓ {wheel.name}")

if __name__ == "__main__":
    build_custom_wheels()
```

Run with:

```bash
python build_custom.py
```

## CI/CD Integration

### GitLab CI

```yaml
# .gitlab-ci.yml
build-wheels:
  image: python:3.11
  script:
    - pip install build
    - python -m pyz3.wheel --all-platforms
  artifacts:
    paths:
      - dist/*.whl
```

### Azure Pipelines

```yaml
# azure-pipelines.yml
steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'

  - script: |
      pip install build
      python -m pyz3.wheel
    displayName: 'Build wheel'

  - task: PublishBuildArtifacts@1
    inputs:
      pathToPublish: 'dist'
```

## Binary Size Optimization

Tips for reducing wheel size:

1. **Use `ReleaseSmall` optimization**:
   ```bash
   export PYDUST_OPTIMIZE=ReleaseSmall
   ```

2. **Strip debug symbols**:
   ```bash
   # Linux/macOS
   strip dist/*.so
   ```

3. **Compress with UPX** (use cautiously):
   ```bash
   upx --best dist/*.so
   ```

4. **Exclude unnecessary files**:
   ```python
   # pyproject.toml
   [tool.setuptools]
   exclude-package-data = {
       "*": ["*.c", "*.h", "*.zig"]
   }
   ```

## Performance Testing

Test wheel performance across platforms:

```python
import time
import myextension

def benchmark():
    start = time.perf_counter()
    for _ in range(1000000):
        myextension.my_function()
    elapsed = time.perf_counter() - start
    print(f"Time: {elapsed:.3f}s")

benchmark()
```

## Resources

- **Zig Cross-compilation**: https://ziglang.org/documentation/master/#Cross-compilation
- **PyPI Publishing**: https://packaging.python.org/tutorials/packaging-projects/
- **Wheel Format**: https://packaging.python.org/specifications/binary-distribution-format/
- **manylinux**: https://github.com/pypa/manylinux

## Support

For issues or questions:
- GitHub Issues: https://github.com/fulcrum-so/pyZ3/issues
- Discussions: https://github.com/fulcrum-so/pyZ3/discussions
