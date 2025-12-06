# Bug Fix: pyz3 init Command TypeError

**Date:** 2025-12-06
**Issue:** TypeError when running `pyz3 init` command
**Status:** ✅ Fixed

## Problem

When running `pyz3 init -n myproject --no-interactive`, users encountered:

```
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

### Error Details

```python
File "/pyz3/init.py", line 106, in init_project_cookiecutter
    cookiecutter(
        str(template_path),
        output_dir=str(output_dir) if output_dir else None,  # ❌ Problem here
        no_input=not use_interactive,
        extra_context=extra_context,
    )
```

### Root Cause

The `output_dir` parameter was being passed as `None` to cookiecutter when initializing a project in the current directory:

```python
output_dir = path.parent if path != Path.cwd() else None  # Returns None
```

Cookiecutter's `output_dir` parameter does not accept `None` - it requires either:
1. A valid path string
2. The parameter to be omitted entirely

## Solution

Modified the cookiecutter invocation to conditionally include the `output_dir` parameter:

### Before (Broken)

```python
output_dir = path.parent if path != Path.cwd() else None

cookiecutter(
    str(template_path),
    output_dir=str(output_dir) if output_dir else None,  # ❌ Passes None
    no_input=not use_interactive,
    extra_context=extra_context,
)
```

### After (Fixed)

```python
# Determine output directory
output_dir = path.parent if path != Path.cwd() else Path.cwd()

# Build cookiecutter kwargs
cookiecutter_kwargs = {
    "no_input": not use_interactive,
    "extra_context": extra_context,
}

# Only specify output_dir if not current directory
if path != Path.cwd():
    cookiecutter_kwargs["output_dir"] = str(output_dir)

cookiecutter(
    str(template_path),
    **cookiecutter_kwargs,  # ✅ Conditionally includes output_dir
)
```

## Testing

### Test Case 1: Initialize in Current Directory

```bash
cd /tmp
pyz3 init -n testproject --description "Test" --email "test@example.com" --no-interactive
```

**Expected Result:** ✅ Creates `/tmp/testproject/` with all files

**Actual Result:** ✅ Works correctly

### Test Case 2: Initialize in Different Directory

```bash
cd /Volumes/ssd
pyz3 init -n myproject --description "My extension" --email "user@example.com" --no-interactive
```

**Expected Result:** ✅ Creates `/Volumes/ssd/myproject/` with all files

**Actual Result:** ✅ Works correctly

### Generated Project Structure

Both test cases correctly generate:

```
myproject/
├── .github/workflows/      # CI/CD
│   ├── ci.yml
│   └── publish.yml
├── .vscode/                # VSCode config
│   ├── extensions.json
│   └── launch.json
├── src/
│   └── myproject.zig       # Zig source
├── myproject/
│   ├── __init__.py         # Python package
│   └── _lib.pyi           # Type stubs
├── test/
│   ├── __init__.py
│   └── test_myproject.py
├── .gitignore
├── build.py
├── LICENSE
├── pyproject.toml
├── README.md
└── renovate.json
```

## Impact

- **Users Affected:** All users running `pyz3 init` or `pyz3 new`
- **Severity:** High (command was completely broken)
- **Fix Complexity:** Low (simple parameter handling fix)

## Files Modified

- `pyz3/init.py` - Lines 102-122
  - Changed output_dir handling
  - Made cookiecutter_kwargs conditional
  - Unpacked kwargs with `**cookiecutter_kwargs`

## Verification

```bash
# Test both command variations
pyz3 init -n test1 --no-interactive
pyz3 new test2

# Both should work without errors
```

## Related Commands

This fix affects:
- `pyz3 init` - Initialize in current directory
- `pyz3 new <name>` - Create new project directory

Both commands now work correctly without the TypeError.

## Lessons Learned

1. **Check API Requirements**: Cookiecutter doesn't accept None for optional parameters
2. **Conditional Parameters**: Use dict unpacking for optional parameters
3. **Test Edge Cases**: Test both "current directory" and "new directory" scenarios
4. **Better Error Messages**: The original error didn't clearly indicate the parameter issue

## Future Improvements

Consider:
- Adding validation for output paths
- Better error messages for cookiecutter failures
- Integration tests for project initialization
- Documentation of cookiecutter template variables

---

**Fixed By:** Repository Maintenance
**Tested:** ✅ Both test cases passing
**Released:** 2025-12-06
