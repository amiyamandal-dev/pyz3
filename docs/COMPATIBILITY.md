# PyZ3 Compatibility Matrix

This document tracks compatibility across Python versions, Zig versions, and operating systems.

## Current Status

**Last Updated:** 2025-12-22
**PyZ3 Version:** 0.x (development)

## Compatibility Table

| Python Version | Zig Version | Linux (x86_64) | Linux (ARM64) | macOS (Intel) | macOS (Apple Silicon) | Windows |
|----------------|-------------|----------------|---------------|---------------|----------------------|---------|
| 3.9            | 0.14.0      | ✅ Tested      | ✅ Tested     | ✅ Tested     | ✅ Tested            | ⚠️ Beta |
| 3.10           | 0.14.0      | ✅ Tested      | ✅ Tested     | ✅ Tested     | ✅ Tested            | ⚠️ Beta |
| 3.11           | 0.14.0      | ✅ Tested      | ✅ Tested     | ✅ Tested     | ✅ Tested            | ⚠️ Beta |
| 3.12           | 0.14.0      | ✅ Tested      | 🔬 Experimental | ✅ Tested   | ✅ Tested            | ❌ Not Supported |
| 3.13           | 0.14.0      | 🔬 Experimental | 🔬 Experimental | 🔬 Experimental | 🔬 Experimental  | ❌ Not Supported |

### Legend

- ✅ **Tested**: Fully tested and supported in production
- ⚠️ **Beta**: Works but has known limitations (see notes below)
- 🔬 **Experimental**: May work but not thoroughly tested
- ❌ **Not Supported**: Known issues, not recommended

## Platform-Specific Notes

### Linux

**Distributions Tested:**
- Ubuntu 20.04, 22.04, 24.04
- Debian 11, 12
- Fedora 38, 39, 40
- Alpine Linux 3.18+ (musl libc)

**Requirements:**
- `python3-dev` package (Ubuntu/Debian)
- `python3-devel` package (Fedora/RHEL)
- GCC or Clang with C11 support

**Known Issues:**
- Alpine Linux: Requires `musl-dev` package
- ARM64: Limited testing on embedded devices

### macOS

**Versions Tested:**
- macOS 12 (Monterey)
- macOS 13 (Ventura)
- macOS 14 (Sonoma)
- macOS 15 (Sequoia)

**Python Sources:**
- ✅ Homebrew Python (`brew install python@3.11`)
- ✅ python.org official installers
- ⚠️ System Python (older macOS versions)
- ✅ pyenv-installed Python

**Known Issues:**
- Framework Python requires special handling (automatic)
- Code signing may be required for distribution

### Windows

**Status:** Beta Support

**Tested Configurations:**
- Windows 10 21H2+
- Windows 11
- Python from python.org
- Mingw-w64 compiler

**Known Issues:**
- Limited API support (`.pyd` extension)
- MSVC compiler not fully tested
- Some tests disabled on Windows

**Requirements:**
- Windows SDK
- Either MSVC or Mingw-w64

## Python Version Specifics

### Python 3.9

**Status:** ✅ Fully Supported

- All features working
- Recommended for production
- Used in CI/CD testing

### Python 3.10

**Status:** ✅ Fully Supported

- All features working
- Recommended for production
- Performance improvements over 3.9

### Python 3.11

**Status:** ✅ Fully Supported

- All features working
- Faster interpreter (10-60% speedup)
- Exception handling improvements

### Python 3.12

**Status:** ⚠️ Beta / 🔬 Experimental

**Working:**
- Core functionality
- Most type conversions
- Basic FFI operations

**Known Issues:**
- Some tests fail due to internal API changes
- PEP 683 (immortal objects) requires special handling
- Performance profiling tools may not work

**Recommendations:**
- Use for development/testing
- Not recommended for production yet
- Report issues to help stabilization

### Python 3.13

**Status:** 🔬 Experimental (Not Released)

**Status:**
- Limited testing (pre-release versions)
- GIL-optional mode not yet supported
- May have breaking changes

**Tracking:**
- PEP 703: Making the GIL Optional
- New C API changes
- Performance monitoring APIs

## Zig Version Requirements

### Zig 0.14.0 (Current)

**Status:** ✅ Recommended

- All features supported
- Stable build system
- Best-tested version

### Zig 0.13.x

**Status:** ❌ Not Supported

- Build system incompatibilities
- Missing required features
- Upgrade to 0.14.0 required

### Zig 0.15.0 (Future)

**Status:** 🔬 Experimental

- Will be supported when released
- May require code updates
- Testing in nightly builds

## Feature Support by Platform

| Feature | Linux | macOS | Windows |
|---------|-------|-------|---------|
| Basic Types (int, str, etc.) | ✅ | ✅ | ✅ |
| Complex Types (dict, list) | ✅ | ✅ | ✅ |
| Native Collections (C-backed) | ✅ | ✅ | ⚠️ |
| SIMD Operations | ✅ | ✅ | ❌ |
| Async/Await | 🔬 | 🔬 | ❌ |
| Limited API (abi3) | ✅ | ✅ | ⚠️ |
| Full API | ✅ | ✅ | ✅ |

## Distribution Support

### PyPI Wheels

**Status:** 🔬 Planning

**Planned:**
- Linux: manylinux2014, manylinux_2_28
- macOS: 11.0+ (Intel & ARM)
- Windows: win_amd64 (experimental)

**Current:**
- Source distribution only
- Requires Zig compiler at install time

### Conda

**Status:** ❌ Not Available

**Future Plans:**
- Conda-forge package
- Requires stable API

## CI/CD Testing

**Automated Testing:**
- ✅ Linux (Ubuntu 22.04): Python 3.9, 3.10, 3.11
- ✅ macOS (latest): Python 3.10, 3.11
- ⚠️ Windows: Limited testing

**Test Frequency:**
- Every pull request
- Nightly builds
- Release candidates

## Reporting Issues

If you encounter platform-specific issues:

1. Check this compatibility matrix
2. Search [existing issues](https://github.com/your-repo/pyz3/issues)
3. Report new issue with:
   - Python version (`python --version`)
   - Zig version (`zig version`)
   - OS and version
   - Error messages
   - Minimal reproduction

## Version Support Policy

**Supported Versions:**
- Latest stable release
- Previous stable release (security fixes only)

**Python Version Support:**
- Follows Python's release cycle
- Drop support 6 months after Python EOL
- Currently: 3.9+ (3.8 EOL Oct 2024)

**Zig Version Support:**
- Latest stable release
- May require updates on new Zig releases
- No support for pre-0.14.0 versions

## Future Roadmap

**Q1 2026:**
- Full Windows support
- Python 3.12 stable
- Pre-built wheels for major platforms

**Q2 2026:**
- Python 3.13 support
- Conda packages
- Performance benchmarks

**Q3 2026:**
- Async/await support
- SIMD on all platforms
- 1.0 release candidate

## References

- [Python Release Schedule](https://peps.python.org/pep-0619/)
- [Zig Release Notes](https://ziglang.org/download/)
- [manylinux Specification](https://github.com/pypa/manylinux)
