# C/C++ Integration with pyz3

pyz3 provides seamless integration with C/C++ code, allowing you to leverage existing libraries and write performance-critical code in C while exposing it to Python through Zig.

## Overview

There are three main ways to integrate C/C++ code with pyz3:

1. **Per-module configuration** - Add C files directly to your extension modules
2. **External dependencies** - Use `pyz3 add` to integrate third-party C/C++ libraries
3. **System libraries** - Link against system-installed libraries (e.g., SQLite, libmath)

## Quick Start

### Method 1: Inline C Files

Add C source files directly to your extension module configuration:

**pyproject.toml:**
```toml
[[tool.pyz3.ext_module]]
name = "mypackage.core"
root = "src/core.zig"
c_sources = ["src/helper.c", "src/utils.c"]
c_include_dirs = ["include/"]
c_flags = ["-O2", "-march=native"]
```

**Project structure:**
```
myproject/
├── src/
│   ├── core.zig        # Zig wrapper
│   ├── helper.c        # C implementation
│   └── utils.c         # More C code
├── include/
│   ├── helper.h
│   └── utils.h
└── pyproject.toml
```

**src/core.zig:**
```zig
const py = @import("pyz3");

const c = @cImport({
    @cInclude("helper.h");
});

pub fn compute(args: struct { x: i32 }) i32 {
    return c.helper_function(args.x);
}

comptime {
    py.rootmodule(@This());
}
```

### Method 2: External Libraries

Use the dependency manager for third-party libraries:

```bash
# Add a C/C++ library
pyz3 add https://github.com/d99kris/rapidcsv

# Use in your module
```

**pyproject.toml:**
```toml
[[tool.pyz3.ext_module]]
name = "mypackage.csv"
root = "src/csv.zig"
link_all_deps = true  # Auto-link all dependencies
```

See [DEPENDENCY_MANAGEMENT.md](./DEPENDENCY_MANAGEMENT.md) for details.

### Method 3: System Libraries

Link against system-installed libraries:

**pyproject.toml:**
```toml
[[tool.pyz3.ext_module]]
name = "mypackage.database"
root = "src/database.zig"
c_libraries = ["sqlite3"]
```

**src/database.zig:**
```zig
const py = @import("pyz3");

const c = @cImport({
    @cInclude("sqlite3.h");
});

pub fn version() !py.PyString {
    const ver = c.sqlite3_libversion();
    return try py.PyString.create(std.mem.span(ver));
}

comptime {
    py.rootmodule(@This());
}
```

## Configuration Reference

### Per-Module Options

All C/C++ integration options are configured in the `[[tool.pyz3.ext_module]]` section:

```toml
[[tool.pyz3.ext_module]]
name = "mypackage.mymodule"
root = "src/mymodule.zig"

# C/C++ source files to compile
c_sources = ["src/file1.c", "src/file2.cpp"]

# Include directories for headers
c_include_dirs = ["include/", "deps/mylib/include"]

# System libraries to link (-l flag)
c_libraries = ["m", "pthread", "sqlite3"]

# C/C++ compiler flags
c_flags = ["-O3", "-march=native", "-DNDEBUG"]

# Linker flags
ld_flags = ["-L/usr/local/lib"]

# Automatically link all dependencies from pyz3_deps.json
link_all_deps = false  # Default: false
```

### Option Details

#### `c_sources`
- **Type:** List of strings
- **Default:** `[]`
- **Description:** C/C++ source files to compile and link with this module
- **Example:** `["src/helper.c", "src/utils.cpp"]`

#### `c_include_dirs`
- **Type:** List of strings
- **Default:** `[]`
- **Description:** Directories to search for header files
- **Example:** `["include/", "deps/rapidcsv/src"]`

#### `c_libraries`
- **Type:** List of strings
- **Default:** `[]`
- **Description:** System libraries to link (equivalent to `-l` flag)
- **Example:** `["m", "pthread", "sqlite3"]`

#### `c_flags`
- **Type:** List of strings
- **Default:** `[]`
- **Description:** Compiler flags for C/C++ sources
- **Example:** `["-O3", "-march=native", "-DNDEBUG"]`

#### `ld_flags`
- **Type:** List of strings
- **Default:** `[]`
- **Description:** Linker flags
- **Example:** `["-L/usr/local/lib"]`

#### `link_all_deps`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Automatically include all dependencies from `pyz3_deps.json`
- **Use case:** When using external libraries added via `pyz3 add`

## Complete Examples

### Example 1: Math Helper in C

Create high-performance math functions in C:

**include/math_helper.h:**
```c
#ifndef MATH_HELPER_H
#define MATH_HELPER_H

double fast_sqrt(double x);
int factorial(int n);

#endif
```

**src/math_helper.c:**
```c
#include "math_helper.h"
#include <math.h>

double fast_sqrt(double x) {
    return sqrt(x);
}

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
```

**src/math.zig:**
```zig
const py = @import("pyz3");
const std = @import("std");

const c = @cImport({
    @cInclude("math_helper.h");
});

pub fn sqrt(args: struct { x: f64 }) !f64 {
    if (args.x < 0) {
        return py.PyException.raise("Cannot take square root of negative number");
    }
    return c.fast_sqrt(args.x);
}

pub fn factorial(args: struct { n: i32 }) !i32 {
    if (args.n < 0) {
        return py.PyException.raise("Factorial requires non-negative integer");
    }
    return c.factorial(@intCast(args.n));
}

comptime {
    py.rootmodule(@This());
}
```

**pyproject.toml:**
```toml
[[tool.pyz3.ext_module]]
name = "mypackage.math"
root = "src/math.zig"
c_sources = ["src/math_helper.c"]
c_include_dirs = ["include/"]
c_libraries = ["m"]  # Link libmath
c_flags = ["-O3", "-ffast-math"]
```

**test_math.py:**
```python
import mypackage.math

def test_sqrt():
    assert mypackage.math.sqrt(9.0) == 3.0
    assert abs(mypackage.math.sqrt(2.0) - 1.414) < 0.001

def test_factorial():
    assert mypackage.math.factorial(5) == 120
    assert mypackage.math.factorial(0) == 1
```

### Example 2: SQLite Wrapper

Wrap SQLite database functionality:

**pyproject.toml:**
```toml
[[tool.pyz3.ext_module]]
name = "mypackage.db"
root = "src/db.zig"
c_libraries = ["sqlite3"]
```

**src/db.zig:**
```zig
const py = @import("pyz3");
const std = @import("std");

const c = @cImport({
    @cInclude("sqlite3.h");
});

pub fn version() !py.PyString {
    const ver = c.sqlite3_libversion();
    return try py.PyString.create(std.mem.span(ver));
}

pub const Database = py.class(struct {
    const Self = @This();

    db: ?*c.sqlite3,

    pub fn __init__(self: *Self, args: struct { path: []const u8 }) !void {
        self.db = null;
        const rc = c.sqlite3_open(args.path.ptr, &self.db);
        if (rc != c.SQLITE_OK) {
            return py.PyException.raise("Failed to open database");
        }
    }

    pub fn __del__(self: *Self) void {
        if (self.db) |db| {
            _ = c.sqlite3_close(db);
        }
    }

    pub fn execute(self: *Self, args: struct { sql: []const u8 }) !void {
        var errmsg: [*c]u8 = undefined;
        const rc = c.sqlite3_exec(
            self.db,
            args.sql.ptr,
            null,
            null,
            &errmsg
        );
        if (rc != c.SQLITE_OK) {
            return py.PyException.raise("SQL error");
        }
    }
});

comptime {
    py.rootmodule(@This());
}
```

### Example 3: CSV Parser with RapidCSV

Use a header-only C++ library:

```bash
pyz3 add https://github.com/d99kris/rapidcsv
```

**pyproject.toml:**
```toml
[[tool.pyz3.ext_module]]
name = "mypackage.csv"
root = "src/csv.zig"
link_all_deps = true
```

**src/csv.zig:**
```zig
const py = @import("pyz3");
const rapidcsv = @import("../bindings/rapidcsv.zig");

// Use rapidcsv.c.* to access the C++ API
// Note: May need C++ wrapper functions for complex types

comptime {
    py.rootmodule(@This());
}
```

## Creating Projects with C Examples

Use the template generator to create projects with C integration:

```bash
# Create new project with C example
pyz3 new myproject

# When prompted:
# include_c_example? Choose "yes"
```

This generates:
- `include/helper.h` - C header
- `src/helper.c` - C implementation
- `src/myproject.zig` - Zig wrapper with example C functions
- `pyproject.toml` - Pre-configured with C integration

## Best Practices

### 1. Header Organization

Keep headers in a dedicated directory:

```
myproject/
├── include/
│   ├── math.h
│   ├── utils.h
│   └── types.h
├── src/
│   ├── math.c
│   ├── utils.c
│   └── wrapper.zig
└── pyproject.toml
```

### 2. Error Handling

Always check return values from C functions:

```zig
pub fn safe_divide(args: struct { a: f64, b: f64 }) !f64 {
    if (args.b == 0.0) {
        return py.PyException.raise("Division by zero");
    }
    const result = c.divide(args.a, args.b);
    if (std.math.isNan(result)) {
        return py.PyException.raise("Invalid result");
    }
    return result;
}
```

### 3. Memory Management

For C code that allocates memory:

```zig
pub fn process_data(args: struct { data: []const u8 }) !py.PyBytes {
    const result = c.process(args.data.ptr, args.data.len);
    defer c.free_result(result);  // Free C memory

    const size = c.get_result_size(result);
    const data_ptr = c.get_result_data(result);

    // Copy to Python bytes (managed by Python GC)
    return try py.PyBytes.from(data_ptr[0..size]);
}
```

### 4. Optimization

Use appropriate compiler flags:

```toml
# For production
c_flags = ["-O3", "-march=native", "-DNDEBUG"]

# For debugging
c_flags = ["-g", "-O0", "-DDEBUG"]
```

### 5. Cross-Platform Support

Be mindful of platform-specific code:

```c
// helper.h
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

EXPORT int my_function(int x);
```

## Troubleshooting

### Header Not Found

**Error:**
```
error: 'myheader.h' file not found
```

**Solution:**
Add the include directory:
```toml
c_include_dirs = ["include/", "src/"]
```

### Undefined Symbol

**Error:**
```
undefined reference to `my_function'
```

**Solution:**
1. Ensure the C file is in `c_sources`
2. Check that the function is not `static` in C
3. For C++, use `extern "C"` linkage

### Linker Errors

**Error:**
```
cannot find -lmylib
```

**Solution:**
1. Install the system library
2. Add library path: `ld_flags = ["-L/usr/local/lib"]`
3. Verify library name: `c_libraries = ["mylib"]`

### C++ Symbols

When using C++ libraries, create C wrapper functions:

**wrapper.cpp:**
```cpp
#include <your_cpp_lib.hpp>

extern "C" {
    void* create_object() {
        return new MyCppClass();
    }

    void destroy_object(void* obj) {
        delete static_cast<MyCppClass*>(obj);
    }

    int call_method(void* obj, int arg) {
        return static_cast<MyCppClass*>(obj)->method(arg);
    }
}
```

**wrapper.h:**
```c
#ifdef __cplusplus
extern "C" {
#endif

void* create_object(void);
void destroy_object(void* obj);
int call_method(void* obj, int arg);

#ifdef __cplusplus
}
#endif
```

## Testing

Test C integration thoroughly:

**test_c_integration.py:**
```python
import pytest
import mypackage

def test_c_function():
    result = mypackage.c_add(2, 3)
    assert result == 5

def test_c_error_handling():
    with pytest.raises(Exception, match="Division by zero"):
        mypackage.divide(10.0, 0.0)

def test_c_memory():
    # Test that C memory is properly managed
    large_data = b"x" * 1000000
    result = mypackage.process(large_data)
    assert len(result) > 0
```

## Performance Considerations

C/C++ integration is ideal for:
- **CPU-intensive computations** - Number crunching, algorithms
- **Existing libraries** - Leverage battle-tested C libraries
- **Low-level operations** - System calls, hardware access
- **Legacy code** - Wrap existing C/C++ codebases

Avoid for:
- **Simple operations** - Zig is often sufficient
- **I/O-bound tasks** - Python's async may be better
- **Frequent Python interactions** - Overhead of crossing language boundary

## See Also

- [DEPENDENCY_MANAGEMENT.md](./DEPENDENCY_MANAGEMENT.md) - External C/C++ libraries
- [Zig @cImport documentation](https://ziglang.org/documentation/master/#cImport)
- [Example: c_integration.zig](../example/c_integration.zig)
- [Test: test_c_integration.py](../test/test_c_integration.py)
