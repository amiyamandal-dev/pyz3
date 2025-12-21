# macOS libPython Linking Workaround

## Problem

On macOS with Framework Python, Zig cannot find the Python library during linking.

Error: `FileNotFound: unable to find dynamic system library 'Python'`

## Root Cause

- **Framework Python**: Library at `/Library/Frameworks/Python.framework/Versions/3.X/Python`
- **Zig expects**: `libPython.dylib` (with `lib` prefix and `.dylib` extension)
- **Mismatch**: Actual file is just `Python` with no prefix/extension

## Quick Fix

Create a symlink:

```bash
# For Python 3.13
sudo ln -s /Library/Frameworks/Python.framework/Versions/3.13/Python \
            /Library/Frameworks/Python.framework/Versions/3.13/lib/libPython.dylib
```

## Automated Solution

Our CI/CD automatically creates this symlink before building on macOS.

See `.github/workflows/build-wheels.yml` for the implementation.

## Alternative

Use Homebrew Python (no workaround needed):

```bash
brew install python@3.13
```

## Verification

```bash
ls -l $(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")/libPython.dylib
```

Should show the symlink pointing to the Python library.
