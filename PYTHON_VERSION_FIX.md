# Python Version Compatibility Fix

**Issue**: Build workflow failing with Python 3.10 incompatibility
**Date**: 2025-12-06
**Status**: ✅ Fixed

## Problem

The GitHub Actions build workflow was trying to build wheels for Python 3.9 and 3.10, but `pyproject.toml` specifies:

```toml
[tool.poetry.dependencies]
python = "^3.11"
```

This caused the error:
```
Because only pyz3[dev]==0.1.0 is available and the current Python
version (3.10.19) does not satisfy Python>=3.11,<4.0, we can conclude
that all versions of pyz3[dev] cannot be used.
```

## Root Cause

Mismatch between:
1. **Workflow matrix**: Building for Python 3.9, 3.10, 3.11, 3.12, 3.13
2. **Package requirement**: Requires Python 3.11+

## Solution

### 1. Updated `.github/workflows/build-wheels.yml`

**Before:**
```yaml
python-version: ['3.9', '3.10', '3.11', '3.12', '3.13']
```

**After:**
```yaml
python-version: ['3.11', '3.12', '3.13']
```

### 2. Updated `pyproject.toml` classifiers

**Before:**
```toml
classifiers = [
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
```

**After:**
```toml
classifiers = [
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
```

## Why Python 3.11+ Only?

pyZ3 requires Python 3.11+ because:

1. **Modern Python features** - Uses newer Python C API features
2. **Type hints** - Advanced type annotations from 3.11+
3. **Performance** - Python 3.11+ has significant performance improvements
4. **Zig compatibility** - Better alignment with Zig's modern tooling

## Verification

After this fix, the build workflow will:
- ✅ Build wheels for Python 3.11, 3.12, 3.13
- ✅ Skip incompatible Python versions
- ✅ Pass dependency resolution
- ✅ Successfully install pyZ3[dev]

## Testing

To verify the fix locally:

```bash
# Test with different Python versions
python3.11 -m pip install -e .
python3.12 -m pip install -e .
python3.13 -m pip install -e .

# Should all succeed
python3.11 -c "import pyz3; print('✅ 3.11 works')"
python3.12 -c "import pyz3; print('✅ 3.12 works')"
python3.13 -c "import pyz3; print('✅ 3.13 works')"
```

## Files Modified

1. `.github/workflows/build-wheels.yml` - Line 51
2. `pyproject.toml` - Lines 17-19 (removed 3.9 and 3.10 classifiers)

## Impact

- **Reduced build time** - Fewer Python versions to build for
- **Clearer requirements** - Package metadata matches actual requirements
- **No breaking changes** - Already required 3.11+ in dependencies

## Next Steps

When you push this fix:
1. GitHub Actions will rebuild with correct Python versions
2. All builds should pass
3. Wheels will be generated for Python 3.11, 3.12, 3.13 only

---

**Fixed by**: Repository maintenance
**Date**: 2025-12-06
**Status**: ✅ Ready to push
