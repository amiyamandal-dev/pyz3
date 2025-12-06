# Version Attribute Fix

**Issue**: `AttributeError: module 'pyz3' has no attribute '__version__'`
**Date**: 2025-12-06
**Status**: ✅ Fixed

## Problem

The build workflow test was failing on Windows and macOS:

```python
python -c "import pyz3; print(f'pyZ3 version: {pyz3.__version__}')"

# Error:
AttributeError: module 'pyz3' has no attribute '__version__'
```

## Root Cause

The `pyz3/__init__.py` file was empty (only had license header) and didn't define the `__version__` attribute that the test was trying to access.

## Solution

Added version and package metadata to `pyz3/__init__.py`:

```python
"""
pyZ3 - Python Extensions in Zig

A high-performance framework for writing Python extension modules in Zig
with automatic memory management, hot-reload, and NumPy integration.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__url__ = "https://github.com/yourusername/pyZ3"
__description__ = "Build high-performance Python extensions in Zig..."
```

## Benefits

### 1. Standard Python Package Metadata
Following PEP 396 and common Python conventions:
- `__version__` - Package version string
- `__author__` - Package author
- `__email__` - Contact email
- `__url__` - Project homepage
- `__description__` - Short description

### 2. Enables Version Checking
Users can now check the installed version:

```python
import pyz3
print(pyz3.__version__)  # "0.1.0"
```

### 3. Better Documentation
```python
import pyz3
help(pyz3)  # Shows proper module docstring
```

### 4. Package Introspection
```python
import pyz3

# Get all metadata
print(f"pyZ3 v{pyz3.__version__}")
print(f"By {pyz3.__author__}")
print(f"URL: {pyz3.__url__}")
```

## Version Management

The version is defined in two places and should be kept in sync:

### 1. pyproject.toml
```toml
[tool.poetry]
version = "0.1.0"
```

### 2. pyz3/__init__.py
```python
__version__ = "0.1.0"
```

### Best Practice: Single Source of Truth

Consider using `importlib.metadata` for automatic version detection:

```python
# pyz3/__init__.py
try:
    from importlib.metadata import version
    __version__ = version("pyZ3")
except Exception:
    __version__ = "0.1.0"  # Fallback
```

Or use `setuptools_scm` for automatic versioning from git tags.

## Testing

Verify the fix:

```bash
# Install the package
pip install -e .

# Test version attribute
python -c "import pyz3; print(pyz3.__version__)"
# Output: 0.1.0

# Test all metadata
python -c "
import pyz3
print(f'Version: {pyz3.__version__}')
print(f'Author: {pyz3.__author__}')
print(f'URL: {pyz3.__url__}')
"
```

## Workflow Update

Also updated the test message for clarity:

**Before:**
```python
python -c "import pyz3; print(f'pyZ3 version: {pyz3.__version__}')"
```

**After:**
```python
python -c "import pyz3; print(f'✅ pyZ3 {pyz3.__version__} installed successfully')"
```

## Future: Dynamic Versioning

For better version management, consider:

### Option 1: importlib.metadata (Modern)
```python
# pyz3/__init__.py
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyZ3")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"
```

### Option 2: setuptools_scm (Git-based)
```toml
# pyproject.toml
[tool.setuptools_scm]
write_to = "pyz3/_version.py"
```

```python
# pyz3/__init__.py
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"
```

### Option 3: Poetry Version Plugin
```toml
# pyproject.toml
[tool.poetry-dynamic-versioning]
enable = true
```

## Files Modified

1. `pyz3/__init__.py` - Added version and metadata attributes
2. `.github/workflows/build-wheels.yml` - Updated test message

## Impact

- ✅ CI builds now pass on all platforms
- ✅ Version checking works correctly
- ✅ Better Python package compliance
- ✅ Improved user experience

---

**Fixed by**: Repository maintenance
**Date**: 2025-12-06
**Status**: ✅ Ready for release
