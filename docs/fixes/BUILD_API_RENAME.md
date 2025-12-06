# Build API Rename - Pydust → pyZ3

**Date**: 2025-12-06
**Status**: ✅ COMPLETE

## Overview

This document summarizes the complete rename of the Zig build API from "Pydust" to "pyZ3" to align with the package rename.

## Problem

The build system API still used "Pydust" naming even though the package was renamed to "pyZ3":

- Structs: `PydustOptions`, `PydustStep`
- Functions: `addPydust()`
- Variables: `pydust_source_file`
- Build steps: `"pydust-test-build"`
- Comments: "Pydust Zig tests"

This created confusion for users who were calling `addPydust()` when the package is named "pyZ3".

## Changes Made

### 1. Struct Renames

**File**: `pyz3/src/pyz3.build.zig`

| Before | After |
|--------|-------|
| `PydustOptions` | `PyZ3Options` |
| `PydustStep` | `PyZ3Step` |

```zig
// Before
pub const PydustOptions = struct {
    test_step: ?*Step = null,
};

pub const PydustStep = struct {
    owner: *std.Build,
    allocator: std.mem.Allocator,
    options: PydustOptions,
    // ...
};

// After
pub const PyZ3Options = struct {
    test_step: ?*Step = null,
};

pub const PyZ3Step = struct {
    owner: *std.Build,
    allocator: std.mem.Allocator,
    options: PyZ3Options,
    // ...
};
```

### 2. Function Renames

**File**: `pyz3/src/pyz3.build.zig`

| Before | After |
|--------|-------|
| `addPydust()` | `addPyZ3()` |

```zig
// Before
pub fn addPydust(b: *std.Build, options: PydustOptions) *PydustStep {
    return PydustStep.add(b, options);
}

// After
pub fn addPyZ3(b: *std.Build, options: PyZ3Options) *PyZ3Step {
    return PyZ3Step.add(b, options);
}
```

### 3. Variable Renames

**File**: `pyz3/src/pyz3.build.zig`

| Before | After |
|--------|-------|
| `pydust_source_file` | `pyz3_source_file` |

```zig
// Before
pub const PydustStep = struct {
    pydust_source_file: []const u8,
    // ...
};

// After
pub const PyZ3Step = struct {
    pyz3_source_file: []const u8,
    // ...
};
```

### 4. Build Step Renames

**File**: `pyz3/src/pyz3.build.zig`

| Before | After |
|--------|-------|
| `"pydust-test-build"` | `"pyz3-test-build"` |

```zig
// Before
const test_build_step = b.step("pydust-test-build", "Build pyz3 test runners");

// After
const test_build_step = b.step("pyz3-test-build", "Build pyZ3 test runners");
```

### 5. Comment Updates

**File**: `pyz3/src/pyz3.build.zig`

```zig
// Before
// Optionally pass your test_step and we will hook up the Pydust Zig tests.

// After
// Optionally pass your test_step and we will hook up the pyZ3 Zig tests.
```

```zig
// Before
/// Configure a Pydust step in the build. From this, you can define Python modules.
/// Adds a pyz3 Python module. The resulting library and test binaries...

// After
/// Configure a pyZ3 step in the build. From this, you can define Python modules.
/// Adds a pyZ3 Python module. The resulting library and test binaries...
```

### 6. User Code Updates

**File**: `pytest.build.zig`

```zig
// Before
const py = @import("./pyz3.build.zig");

const pyz3 = py.addPydust(b, .{
    .test_step = test_step,
});

// After
const py = @import("./pyz3/src/pyz3.build.zig");

const pyz3 = py.addPyZ3(b, .{
    .test_step = test_step,
});
```

### 7. Duplicate File Removed

**Removed**: `/Volumes/ssd/ziggy-pydust/pyz3.build.zig` (root duplicate)

**Kept**: `/Volumes/ssd/ziggy-pydust/pyz3/src/pyz3.build.zig` (source)

The root `pyz3.build.zig` was a duplicate of the source file. It has been removed, and all imports now point to the correct source location `pyz3/src/pyz3.build.zig`.

## Files Modified

| File | Changes |
|------|---------|
| `pyz3/src/pyz3.build.zig` | 9 renames (structs, functions, variables, steps, comments) |
| `pytest.build.zig` | Updated import path and function call |
| `pyz3.build.zig` (root) | Removed (duplicate) |

## Impact Analysis

### Breaking Changes

⚠️ **This is a breaking change** for all existing user build files.

**Before** (old API):
```zig
const py = @import("./pyz3.build.zig");

pub fn build(b: *std.Build) void {
    const pyz3 = py.addPydust(b, .{
        .test_step = test_step,
    });

    _ = pyz3.addPythonModule(.{ ... });
}
```

**After** (new API):
```zig
const py = @import("./pyz3/src/pyz3.build.zig");

pub fn build(b: *std.Build) void {
    const pyz3 = py.addPyZ3(b, .{
        .test_step = test_step,
    });

    _ = pyz3.addPythonModule(.{ ... });
}
```

### Migration Guide

Users need to update their `build.zig` files:

1. **Update import path**:
   ```zig
   // Old
   const py = @import("./pyz3.build.zig");

   // New
   const py = @import("./pyz3/src/pyz3.build.zig");
   ```

2. **Update function call**:
   ```zig
   // Old
   const pyz3 = py.addPydust(b, .{ ... });

   // New
   const pyz3 = py.addPyZ3(b, .{ ... });
   ```

3. **No other changes needed** - The rest of the API remains the same:
   - `pyz3.addPythonModule()` - unchanged
   - `PythonModuleOptions` - unchanged
   - All other build functionality - unchanged

## Verification

### Build Test
```bash
$ zig build
# Result: BUILD SUCCESSFUL
```

All builds complete successfully with the new API naming.

## Summary

### What Changed
- ✅ All "Pydust" naming → "pyZ3" in build API
- ✅ Import path updated from root to source location
- ✅ Duplicate file removed
- ✅ All builds successful

### What Stayed the Same
- ✅ `PythonModuleOptions` (unchanged)
- ✅ `addPythonModule()` method (unchanged)
- ✅ Build functionality (unchanged)
- ✅ Python module configuration (unchanged)

### Naming Consistency Achieved

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| Package name | ziggy-pydust | pyZ3 | ✅ Consistent |
| Import name | pydust | pyz3 | ✅ Consistent |
| Build API structs | Pydust* | PyZ3* | ✅ Consistent |
| Build API functions | addPydust | addPyZ3 | ✅ Consistent |
| Build step names | pydust-* | pyz3-* | ✅ Consistent |

## Result

The rename from "Pydust" to "pyZ3" is now **100% complete** across:
- ✅ Package metadata
- ✅ Python code
- ✅ Zig source code
- ✅ Build system API
- ✅ GitHub workflows
- ✅ Templates
- ✅ Documentation

All references to "Pydust" have been removed from the codebase (except in historical documentation).

---

**Completed**: 2025-12-06
**Build Status**: ✅ Passing
**Breaking Change**: Yes - requires user build.zig updates
**Migration Difficulty**: Low - 2 simple changes
