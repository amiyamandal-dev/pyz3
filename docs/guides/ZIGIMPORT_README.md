# zigimport - Import Zig Directly from Python!

**zigimport** allows you to import `.zig` files directly from Python without manual compilation, just like `rustimport` for Rust.

## Quick Start

**1. Create a Zig module** (`my_module.zig`):

```zig
const py = @import("pyz3");

pub fn add(args: struct { a: i64, b: i64 }) i64 {
    return args.a + args.b;
}

pub fn greet(args: struct { name: []const u8 }) []const u8 {
    return "Hello, " ++ args.name ++ "!";
}

comptime {
    py.rootmodule(@This());
}
```

**2. Import and use from Python**:

```python
import pyz3.zigimport  # Installs the import hook
import my_module        # Automatically compiles my_module.zig!

print(my_module.add(5, 3))           # Output: 8
print(my_module.greet("World"))      # Output: Hello, World!
```

That's it! No `zig build`, no `build.zig`, no setup required!

## How It Works

1. **First import**: zigimport finds `my_module.zig` and compiles it using pyz3's build system
2. **Caching**: The compiled `.so` file is cached in `~/.zigimport/`
3. **Change detection**: Only recompiles if the `.zig` file is modified (checks file mtime)
4. **Subsequent imports**: Uses the cached version (instant!)

## Features

- ✨ **Zero Configuration** - Just import and go
- ⚡ **Smart Caching** - Only recompiles when source changes
- 🔧 **Configurable** - Control optimization level via environment variables
- 📊 **Jupyter Support** - Works great in notebooks
- 🛠️ **pyz3 Integration** - Uses pyz3's proven build system
- 🎯 **Simple** - Clean, focused implementation (~260 lines)

## Configuration

Control zigimport behavior with environment variables:

```python
import os

# Set optimization level
os.environ["ZIGIMPORT_OPTIMIZE"] = "ReleaseFast"  # Debug|ReleaseSafe|ReleaseFast|ReleaseSmall

# Enable verbose output
os.environ["ZIGIMPORT_VERBOSE"] = "1"

# Force rebuild on every import (for debugging)
os.environ["ZIGIMPORT_FORCE_REBUILD"] = "1"

# Custom cache directory
os.environ["ZIGIMPORT_BUILD_DIR"] = "/tmp/my_zig_cache"

# Now import
import pyz3.zigimport
import my_module
```

## CLI Commands

Manage zigimport from command line:

```bash
# Show help
pyz3 import

# Enable automatic import
pyz3 import enable

# Disable automatic import
pyz3 import disable

# Clear compilation cache
pyz3 import clear-cache

# Show configuration
pyz3 import info
```

## Jupyter/IPython Support

Use in Jupyter notebooks:

```python
# Load the extension
%load_ext pyz3.import_hook

# Now import .zig files
import my_module
my_module.add(10, 20)
```

## Examples

### Example 1: Simple Math

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
import pyz3.zigimport
import math_utils

print(math_utils.factorial(5))     # 120
print(math_utils.is_prime(17))     # True
```

### Example 2: With NumPy

**File: `array_ops.zig`**

```zig
const py = @import("pyz3");

pub fn sum_array(args: struct { arr: py.PyArray(@This()) }) !f64 {
    const data = try args.arr.asSlice(f64);
    var total: f64 = 0.0;
    for (data) |val| {
        total += val;
    }
    return total;
}

comptime {
    py.rootmodule(@This());
}
```

**Usage:**

```python
import pyz3.zigimport
import numpy as np
import array_ops

arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(array_ops.sum_array(arr))  # 15.0
```

### Example 3: Development Workflow

```python
import os
os.environ["ZIGIMPORT_VERBOSE"] = "1"
os.environ["ZIGIMPORT_OPTIMIZE"] = "Debug"

import pyz3.zigimport
import my_module

# Use the module
result = my_module.process(data)

# Edit my_module.zig...

# Reload to see changes
import importlib
importlib.reload(my_module)  # Auto-recompiles if changed!
```

## Comparison with Traditional Workflow

### Traditional pyz3:

```bash
# 1. Create Zig code
$ cat > my_module.zig

# 2. Create build.zig or pyproject.toml config
$ cat > build.zig

# 3. Build manually
$ zig build

# 4. Finally import
$ python -c "import my_module"
```

### With zigimport:

```bash
# 1. Create Zig code
$ cat > my_module.zig

# 2. Import directly!
$ python -c "import pyz3.zigimport; import my_module"
```

## Implementation Details

- **~260 lines** of clean, focused Python code
- Uses pyz3's existing `buildzig.zig_build()` function
- Leverages `ExtModule` and `ToolPydust` configurations
- Simple mtime-based change detection
- No external dependencies beyond pyz3

## Architecture

```
User Code:
  import pyz3.zigimport    →  Installs import hook
  import my_module         →  Triggers compilation

ZigImportFinder:
  1. Find my_module.zig in sys.path
  2. Check cache: needs rebuild?
     - Check file mtime
     - Check if .so exists
  3. If rebuild needed:
     - Create ExtModule config
     - Call buildzig.zig_build()
     - Move .so to cache
  4. Load compiled .so
  5. Return module to Python
```

## Limitations

1. **Simple Modules Only**: Best for single-file modules; complex builds need traditional `build.zig`
2. **No Dependency Tracking**: Only tracks the main .zig file, not its imports
3. **Host Platform**: Compiles for current platform only

## Performance

| Operation | Time |
|-----------|------|
| First import (compilation) | 1-5 seconds |
| Cached import (no changes) | ~5-10ms |
| Import after edit | 1-5 seconds (recompile) |

## Troubleshooting

**Module not found?**
- Ensure `.zig` file is in current directory or in `sys.path`
- Check filename matches import name: `import foo` → `foo.zig`

**Compilation errors?**
- Enable verbose mode: `os.environ["ZIGIMPORT_VERBOSE"] = "1"`
- Check error messages from zig build
- Fix your .zig code and import again

**Stale cache?**
```bash
pyz3 import clear-cache
```

Or force rebuild:
```python
os.environ["ZIGIMPORT_FORCE_REBUILD"] = "1"
```

## Credits

Inspired by:
- [rustimport](https://github.com/mityax/rustimport) - Import Rust from Python
- [cppimport](https://github.com/tbenthompson/cppimport) - Import C++ from Python

## License

Apache License 2.0 (same as pyz3)
