# zigimport - Import Zig Directly from Python!

**zigimport** is pyz3's automatic import system that allows you to import `.zig` files directly from Python without any manual compilation steps.

Heavily inspired by [rustimport](https://github.com/mityax/rustimport) and [cppimport](https://github.com/tbenthompson/cppimport).

## Features

- 🚀 **Zero Manual Compilation** - Just `import my_module` and it compiles automatically
- ⚡ **Smart Change Detection** - Only recompiles when source files change (checksum-based)
- 📦 **Automatic Caching** - Compiled modules are cached for instant subsequent imports
- 🔧 **Production Mode** - Skip checksum validation for faster production imports
- 📊 **Incremental Builds** - Fast recompilation using Zig's incremental compilation
- 🎓 **Jupyter/IPython Support** - Works seamlessly in notebooks
- ⚙️ **Configurable** - Control optimization level, verbosity, and more via environment variables

## Quick Start

### Basic Usage

**1. Create a simple Zig module** (`my_module.zig`):

```zig
const py = @import("pyz3");

pub fn hello(args: struct { name: []const u8 }) ![]const u8 {
    return "Hello, " ++ args.name ++ "!";
}

pub fn add(args: struct { a: i64, b: i64 }) i64 {
    return args.a + args.b;
}

comptime {
    py.rootmodule(@This());
}
```

**2. Import and use it from Python**:

```python
import zigimport  # Automatically installs the import hook
import my_module  # Automatically compiles my_module.zig

# Use it!
print(my_module.hello("World"))  # Output: Hello, World!
print(my_module.add(5, 3))        # Output: 8
```

That's it! No manual `zig build`, no compilation step, no setup.py!

### How It Works

1. **First import**: zigimport compiles `my_module.zig` to a Python extension
2. **Subsequent imports**: zigimport checks if the source changed (via SHA256 checksum)
3. **If unchanged**: Uses cached compiled module (instant)
4. **If changed**: Recompiles automatically

## Jupyter/IPython Notebooks

zigimport works great in Jupyter notebooks:

```python
# In a Jupyter cell:
%load_ext pyz3.import_hook

# Now you can import .zig files:
import my_module
my_module.hello("Jupyter")
```

## Configuration

Configure zigimport behavior via environment variables:

### `ZIGIMPORT_OPTIMIZE`

Set optimization level for compiled modules:

```bash
export ZIGIMPORT_OPTIMIZE=ReleaseFast  # Options: Debug, ReleaseSafe, ReleaseFast, ReleaseSmall
```

```python
import os
os.environ["ZIGIMPORT_OPTIMIZE"] = "ReleaseFast"
import zigimport
import my_module  # Compiled with ReleaseFast optimization
```

### `ZIGIMPORT_RELEASE_MODE`

Skip checksum validation in production for faster imports:

```bash
export ZIGIMPORT_RELEASE_MODE=1
```

```python
import os
os.environ["ZIGIMPORT_RELEASE_MODE"] = "1"
import zigimport
import my_module  # Skips checksum check, uses existing .so
```

### `ZIGIMPORT_VERBOSE`

Enable verbose output to see what zigimport is doing:

```bash
export ZIGIMPORT_VERBOSE=1
```

```python
import os
os.environ["ZIGIMPORT_VERBOSE"] = "1"
import zigimport
import my_module
# Output:
# [zigimport] Found Zig source: /path/to/my_module.zig for module my_module
# [zigimport] Checksum match for my_module: using cached build
```

### `ZIGIMPORT_FORCE_REBUILD`

Force recompilation on every import (useful for debugging):

```bash
export ZIGIMPORT_FORCE_REBUILD=1
```

### `ZIGIMPORT_BUILD_DIR`

Custom build/cache directory (default: `~/.zigimport`):

```bash
export ZIGIMPORT_BUILD_DIR=/tmp/my_zig_cache
```

### `ZIGIMPORT_PYTHON`

Specify Python executable to use:

```bash
export ZIGIMPORT_PYTHON=/usr/bin/python3.11
```

### `ZIGIMPORT_ZIG`

Specify Zig executable to use:

```bash
export ZIGIMPORT_ZIG=/usr/local/bin/zig
```

### `ZIGIMPORT_INCLUDE_PATHS`

Additional search paths for .zig files (colon-separated):

```bash
export ZIGIMPORT_INCLUDE_PATHS=/path/to/modules:/another/path
```

## CLI Commands

Manage zigimport from the command line:

### Enable/Disable

```bash
# Enable automatic import
pyz3 import enable

# Disable automatic import
pyz3 import disable
```

### Clear Cache

```bash
# Clear compiled module cache
pyz3 import clear-cache
```

### Show Configuration

```bash
# Display current configuration
pyz3 import info

# Output:
# zigimport Configuration:
#   Status: Enabled
#   Build directory: /home/user/.zigimport
#   Optimization: Debug
#   Python executable: /usr/bin/python3
#   Zig executable: /usr/bin/zig
#   Release mode: False
#   Verbose: False
#   Force rebuild: False
```

## Advanced Usage

### Programmatic Control

```python
import zigimport

# Manually install/uninstall
zigimport.install()    # Enable import hook
zigimport.uninstall()  # Disable import hook

# Force rebuild all modules
zigimport.force_rebuild()

# Clear cache
zigimport.clear_cache()
```

### Development Workflow

During development, use verbose mode to see what's happening:

```python
import os
os.environ["ZIGIMPORT_VERBOSE"] = "1"
os.environ["ZIGIMPORT_OPTIMIZE"] = "Debug"

import zigimport
import my_module

# Edit my_module.zig...

# Reload the module
import importlib
importlib.reload(my_module)  # Automatically recompiles if changed
```

### Production Deployment

For production, enable release mode:

```python
import os
os.environ["ZIGIMPORT_RELEASE_MODE"] = "1"
os.environ["ZIGIMPORT_OPTIMIZE"] = "ReleaseFast"

import zigimport
import my_module  # Fast import, no checksum validation
```

## Module Search Path

zigimport searches for `.zig` files in:

1. Paths in `sys.path`
2. Current working directory
3. Custom paths in `ZIGIMPORT_INCLUDE_PATHS`

### Module Name Resolution

```
import my_module        # Looks for: my_module.zig
import pkg.my_module    # Looks for: pkg/my_module.zig or pkg/my_module/__init__.zig
```

## Examples

### Example 1: Simple Math Module

**File: `math_utils.zig`**

```zig
const py = @import("pyz3");

pub fn factorial(args: struct { n: u64 }) u64 {
    if (args.n <= 1) return 1;
    return args.n * factorial(.{ .n = args.n - 1 });
}

pub fn is_prime(args: struct { n: u64 }) bool {
    if (args.n < 2) return false;
    if (args.n == 2) return true;
    if (args.n % 2 == 0) return false;

    var i: u64 = 3;
    while (i * i <= args.n) : (i += 2) {
        if (args.n % i == 0) return false;
    }
    return true;
}

comptime {
    py.rootmodule(@This());
}
```

**Usage:**

```python
import zigimport
import math_utils

print(math_utils.factorial(5))     # 120
print(math_utils.is_prime(17))     # True
print(math_utils.is_prime(18))     # False
```

### Example 2: String Processing

**File: `text_utils.zig`**

```zig
const py = @import("pyz3");
const std = @import("std");

pub fn reverse_string(args: struct { s: []const u8 }) ![]const u8 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var reversed = try allocator.alloc(u8, args.s.len);
    var i: usize = 0;
    while (i < args.s.len) : (i += 1) {
        reversed[i] = args.s[args.s.len - 1 - i];
    }
    return reversed;
}

pub fn count_vowels(args: struct { s: []const u8 }) u32 {
    var count: u32 = 0;
    for (args.s) |c| {
        switch (c) {
            'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U' => count += 1,
            else => {},
        }
    }
    return count;
}

comptime {
    py.rootmodule(@This());
}
```

**Usage:**

```python
import zigimport
import text_utils

print(text_utils.reverse_string("hello"))    # "olleh"
print(text_utils.count_vowels("beautiful"))  # 5
```

### Example 3: NumPy Integration

**File: `array_ops.zig`**

```zig
const py = @import("pyz3");

pub fn sum_array(args: struct { arr: py.PyArray(@This()) }) !f64 {
    const data = try args.arr.asSlice(f64);
    var sum: f64 = 0.0;
    for (data) |val| {
        sum += val;
    }
    return sum;
}

pub fn multiply_by_scalar(args: struct {
    arr: py.PyArray(@This()),
    scalar: f64,
}) !py.PyArray(@This()) {
    const data = try args.arr.asSliceMut(f64);
    for (data) |*val| {
        val.* *= args.scalar;
    }
    return args.arr;
}

comptime {
    py.rootmodule(@This());
}
```

**Usage:**

```python
import zigimport
import numpy as np
import array_ops

arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print(array_ops.sum_array(arr))              # 15.0

array_ops.multiply_by_scalar(arr, 2.0)
print(arr)                                    # [2. 4. 6. 8. 10.]
```

## Comparison with Traditional Workflow

### Traditional pyz3 workflow:

```bash
# 1. Write Zig code
$ cat > my_module.zig

# 2. Create build.zig
$ cat > build.zig

# 3. Build
$ zig build

# 4. Now you can import
$ python
>>> import my_module
```

### With zigimport:

```bash
# 1. Write Zig code
$ cat > my_module.zig

# 2. Import directly!
$ python
>>> import zigimport
>>> import my_module  # Automatically compiles!
```

## Performance

- **First import**: Compilation time (depends on module complexity)
- **Subsequent imports**: ~1-5ms (checksum calculation + cache lookup)
- **Release mode**: ~0.1ms (no checksum, just existence check)

### Benchmarks

```python
import time
import os

# Development mode
os.environ["ZIGIMPORT_OPTIMIZE"] = "Debug"
import zigimport
start = time.time()
import my_module
print(f"First import: {(time.time() - start) * 1000:.2f}ms")  # ~1000-5000ms

# Subsequent import
import importlib
importlib.invalidate_caches()
start = time.time()
importlib.reload(my_module)
print(f"Cached import: {(time.time() - start) * 1000:.2f}ms")  # ~2-5ms

# Release mode
os.environ["ZIGIMPORT_RELEASE_MODE"] = "1"
importlib.invalidate_caches()
start = time.time()
importlib.reload(my_module)
print(f"Release mode: {(time.time() - start) * 1000:.2f}ms")  # ~0.1ms
```

## Troubleshooting

### Module Not Found

**Problem**: `ModuleNotFoundError: No module named 'my_module'`

**Solution**: Make sure:
1. `my_module.zig` exists in your current directory or Python path
2. The import hook is installed: `import zigimport` before importing your module
3. Check search paths with `ZIGIMPORT_INCLUDE_PATHS`

### Compilation Errors

**Problem**: Import fails with compilation errors

**Solution**:
1. Enable verbose mode: `os.environ["ZIGIMPORT_VERBOSE"] = "1"`
2. Check the error message for Zig compilation issues
3. Fix your .zig code
4. The module will automatically recompile on next import

### Stale Cache

**Problem**: Changes to .zig file not reflected

**Solution**:
```bash
# Clear the cache
pyz3 import clear-cache

# Or force rebuild
export ZIGIMPORT_FORCE_REBUILD=1
```

### Permission Errors

**Problem**: Permission denied on cache directory

**Solution**:
```bash
# Use custom cache directory
export ZIGIMPORT_BUILD_DIR=/tmp/my_cache
```

## Limitations

1. **Complex Dependencies**: If your .zig file imports other local .zig files, zigimport currently only tracks changes to the main file
2. **Build Configuration**: Limited to simple modules; complex build configurations may need traditional `build.zig`
3. **Cross-compilation**: Currently compiles for the host platform only

## Future Enhancements

Planned features:

- [ ] Dependency tracking for imported .zig files
- [ ] Custom build configuration support
- [ ] Parallel compilation of multiple modules
- [ ] Integration with pyz3's existing build system
- [ ] Watch mode for auto-reload during development

## Acknowledgments

zigimport is inspired by:
- [rustimport](https://github.com/mityax/rustimport) - Import Rust directly from Python
- [cppimport](https://github.com/tbenthompson/cppimport) - Import C++ directly from Python
- [maturin-import-hook](https://github.com/PyO3/maturin-import-hook) - Official PyO3 import hook

## License

Apache License 2.0 (same as pyz3)
