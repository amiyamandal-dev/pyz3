# C/C++ Dependency Management

pyZ3 includes powerful C/C++ dependency management that automatically generates Zig bindings and integrates external libraries into your project.

## Overview

The `pyz3 add` command allows you to:
1. **Clone** C/C++ libraries from GitHub or use local paths
2. **Auto-generate** Zig bindings using `@cImport`
3. **Track versions** and dependency metadata
4. **Integrate** seamlessly into your build system
5. **Generate** Python wrapper templates

This makes it easy to leverage existing C/C++ libraries in your pyZ3 extensions!

## Quick Start

```bash
# Add a C/C++ library from GitHub
pyz3 add https://github.com/d99kris/rapidcsv

# Add a local library
pyz3 add /path/to/mylibrary

# List dependencies
pyz3 list

# Remove a dependency
pyz3 remove rapidcsv
```

## Commands

### `pyz3 add`

Add a C/C++ library to your project.

```bash
pyz3 add <source> [options]
```

**Arguments:**
- `<source>` - GitHub URL, Git URL, or local filesystem path

**Options:**
- `-n, --name <NAME>` - Override dependency name (defaults to repo/directory name)
- `--headers <HEADERS...>` - Specify main headers (auto-detected if not provided)
- `-v, --verbose` - Enable verbose output

**Examples:**

```bash
# Add from GitHub
pyz3 add https://github.com/nlohmann/json

# Add with custom name
pyz3 add https://github.com/nothings/stb --name stb_image

# Add local library
pyz3 add ../mylib

# Specify headers explicitly
pyz3 add https://github.com/libuv/libuv --headers include/uv.h
```

### `pyz3 list`

List all C/C++ dependencies in the project.

```bash
pyz3 list
```

**Output:**
```
Dependencies (2):

  • rapidcsv
    Version: v8.90
    Source: https://github.com/d99kris/rapidcsv
    Type: Header-only
    Headers: rapidcsv.h

  • json
    Version: v3.11.3
    Source: https://github.com/nlohmann/json
    Type: Header-only
    Headers: json.hpp
```

### `pyz3 remove`

Remove a C/C++ dependency from the project.

```bash
pyz3 remove <name>
```

**Example:**
```bash
pyz3 remove rapidcsv
```

## What Gets Created

When you run `pyz3 add`, the following files are generated:

### 1. Dependency Tracking (`pyz3_deps.json`)

JSON file tracking all dependencies:

```json
{
  "rapidcsv": {
    "name": "rapidcsv",
    "source": "https://github.com/d99kris/rapidcsv",
    "version": "v8.90",
    "include_dirs": ["deps/rapidcsv/src"],
    "headers": ["deps/rapidcsv/src/rapidcsv.h"],
    "is_header_only": true
  }
}
```

### 2. Cloned Source (`deps/<name>/`)

The library source code:

```
deps/
└── rapidcsv/
    ├── src/
    │   └── rapidcsv.h
    ├── README.md
    └── ...
```

### 3. Zig Bindings (`bindings/<name>.zig`)

Auto-generated Zig bindings:

```zig
// Auto-generated Zig bindings for rapidcsv
// Source: https://github.com/d99kris/rapidcsv
// Version: v8.90

// Import the C library
pub const c = @cImport({
    @cInclude("rapidcsv.h");
});

// Re-export commonly used types and functions
// TODO: Add convenience wrappers for common operations
```

### 4. Build Configuration (`bindings/deps.zig.inc`)

Build system integration:

```zig
// Auto-generated dependency configuration
const std = @import("std");

pub fn addDependencies(b: *std.Build, target: std.Build.ResolvedTarget, lib: anytype) void {
    // Dependency: rapidcsv
    lib.addIncludePath(b.path("deps/rapidcsv/src"));
    lib.linkLibC();
}
```

### 5. Python Wrapper Template (`src/<name>_wrapper.zig`)

Template for exposing the library to Python:

```zig
// Python wrapper for rapidcsv
const py = @import("pyz3");
const rapidcsv = @import("../bindings/rapidcsv.zig");

// Example: Expose a C function to Python
pub fn read_csv(args: struct { filename: []const u8 }) !py.PyString {
    // Use rapidcsv.c.* here
    return py.PyString.create("CSV data");
}

comptime {
    py.rootmodule(@This());
}
```

## Integration Guide

### Step 1: Add a Dependency

```bash
pyz3 add https://github.com/d99kris/rapidcsv
```

### Step 2: Update Your Build System

Add to your `build.zig`:

```zig
const std = @import("std");
const deps = @import("bindings/deps.zig.inc");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create your library
    const lib = b.addSharedLibrary(.{
        .name = "mymodule",
        .root_source_file = b.path("src/mymodule.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add C/C++ dependencies
    deps.addDependencies(b, target, lib);

    b.installArtifact(lib);
}
```

### Step 3: Use in Your Zig Code

Import and use the bindings:

```zig
const py = @import("pyz3");
const rapidcsv = @import("../bindings/rapidcsv.zig");

pub fn read_csv_file(args: struct { path: []const u8 }) !void {
    // Use the C++ library through Zig bindings
    // Note: rapidcsv is C++, so you'll need to create C wrappers
    // or use Zig's C++ interop features
}

comptime {
    py.rootmodule(@This());
}
```

### Step 4: Build and Test

```bash
pyz3 develop
pytest
```

## Advanced Usage

### Header-Only Libraries

For header-only C++ libraries (like RapidCSV, nlohmann/json):

```bash
pyz3 add https://github.com/nlohmann/json
```

No compilation needed - just include the headers!

### Libraries with Source Files

For libraries that need compilation:

```bash
pyz3 add https://github.com/libuv/libuv
```

pyZ3 will:
- Detect `.c` and `.cpp` files
- Add them to the build configuration
- Link the compiled library

### System Libraries

You can also reference system libraries:

```bash
# Just create bindings for system SQLite
pyz3 add /usr/include/sqlite3.h --name sqlite3
```

### Custom Headers

Specify which headers to expose:

```bash
pyz3 add https://github.com/stb/stb \
  --headers stb_image.h stb_image_write.h
```

## Real-World Examples

### Example 1: CSV Processing with RapidCSV

```bash
# Add the library
pyz3 add https://github.com/d99kris/rapidcsv
```

Create `src/csv_module.zig`:

```zig
const py = @import("pyz3");
const rapidcsv = @import("../bindings/rapidcsv.zig");

pub fn parse_csv(args: struct {
    data: []const u8
}) !py.PyList {
    // Parse CSV and return Python list
    // Implementation here...
    return py.PyList.new();
}

comptime {
    py.rootmodule(@This());
}
```

### Example 2: JSON Processing with nlohmann/json

```bash
# Add the library
pyz3 add https://github.com/nlohmann/json
```

Create C++ wrapper (since nlohmann/json is C++):

```cpp
// src/json_wrapper.cpp
#include <json.hpp>
extern "C" {
    const char* parse_json(const char* input) {
        auto j = nlohmann::json::parse(input);
        return j.dump().c_str();
    }
}
```

Use in Zig:

```zig
const py = @import("pyz3");
const c = @cImport({
    @cInclude("json_wrapper.h");
});

pub fn parse_json(args: struct { data: []const u8 }) !py.PyString {
    const result = c.parse_json(args.data.ptr);
    return py.PyString.create(std.mem.span(result));
}

comptime {
    py.rootmodule(@This());
}
```

### Example 3: Image Processing with stb_image

```bash
# Add stb_image
pyz3 add https://github.com/nothings/stb \
  --headers stb_image.h
```

Use in Zig:

```zig
const py = @import("pyz3");
const stb = @import("../bindings/stb.zig");

pub fn load_image(args: struct {
    filename: []const u8
}) !py.PyBytes {
    var width: c_int = undefined;
    var height: c_int = undefined;
    var channels: c_int = undefined;

    const data = stb.c.stbi_load(
        args.filename.ptr,
        &width,
        &height,
        &channels,
        0
    );

    defer stb.c.stbi_image_free(data);

    // Convert to Python bytes
    const size = @intCast(usize, width * height * channels);
    return py.PyBytes.from(data[0..size]);
}

comptime {
    py.rootmodule(@This());
}
```

## Dependency File Format

The `pyz3_deps.json` file stores dependency metadata:

```json
{
  "library_name": {
    "name": "library_name",
    "source": "https://github.com/user/repo",
    "version": "v1.0.0",
    "include_dirs": ["deps/library_name/include"],
    "lib_dirs": ["deps/library_name/lib"],
    "libraries": ["mylibrary"],
    "headers": ["deps/library_name/include/mylibrary.h"],
    "cflags": ["-DSOME_FLAG"],
    "ldflags": ["-pthread"],
    "source_files": ["deps/library_name/src/file.c"],
    "is_header_only": false
  }
}
```

## Troubleshooting

### "Header not found" Errors

If Zig can't find headers:

1. Check `include_dirs` in `pyz3_deps.json`
2. Manually specify headers:
   ```bash
   pyz3 add <source> --headers path/to/header.h
   ```
3. Update `bindings/deps.zig.inc` if needed

### C++ Libraries

For C++ libraries, you need C wrappers:

1. Create a C wrapper file (`wrapper.cpp`)
2. Expose C functions with `extern "C"`
3. Add wrapper to build system
4. Import in Zig using `@cImport`

Example wrapper:

```cpp
// src/cpp_wrapper.cpp
#include <your_cpp_library.hpp>

extern "C" {
    void* create_object() {
        return new YourClass();
    }

    void destroy_object(void* obj) {
        delete static_cast<YourClass*>(obj);
    }
}
```

### Version Conflicts

If dependencies conflict:

1. Check versions: `pyz3 list`
2. Remove conflicting deps: `pyz3 remove <name>`
3. Add specific version (clone specific tag/branch)

### Build Errors

If build fails:

1. Check `bindings/deps.zig.inc`
2. Verify include paths
3. Ensure C/C++ files compile
4. Use `pyz3 develop --verbose`

## Best Practices

### 1. Document Dependencies

Add to your `README.md`:

```markdown
## C/C++ Dependencies

This project uses the following C/C++ libraries:
- [RapidCSV](https://github.com/d99kris/rapidcsv) - CSV parsing
- [nlohmann/json](https://github.com/nlohmann/json) - JSON processing
```

### 2. Pin Versions

After adding, note the version:

```bash
pyz3 list  # Shows: Version: v8.90
```

Document in your README which versions are tested.

### 3. Test Thoroughly

Create tests for C/C++ integration:

```python
# tests/test_csv.py
import mymodule

def test_csv_parsing():
    result = mymodule.parse_csv("a,b,c\n1,2,3")
    assert result == [["1", "2", "3"]]
```

### 4. Handle Errors

C libraries may return errors - handle them:

```zig
pub fn safe_function(args: struct { value: i32 }) !i32 {
    const result = c_library.c.some_function(args.value);
    if (result < 0) {
        return py.PyException.raise("LibraryError: Operation failed");
    }
    return result;
}
```

## Comparison with Other Tools

| Feature | pyZ3 | Cargo (Rust) | pip (Python) |
|---------|--------|--------------|--------------|
| Language | C/C++ → Zig | Rust | Python |
| Auto-bindings | ✅ Yes | Manual | N/A |
| Version tracking | ✅ Yes | ✅ Yes | ✅ Yes |
| Build integration | ✅ Auto | ✅ Auto | Manual |
| Cross-compilation | ✅ Yes (Zig) | ✅ Yes | Limited |

## See Also

- [CLI Reference](./CLI.md) - All CLI commands
- [Zig `@cImport` docs](https://ziglang.org/documentation/master/#cImport) - How Zig imports C
- [pyZ3 Documentation](https://pyz3.fulcrum.so) - Main docs
- [Examples](../example/) - Sample code

## Feedback

Have ideas for improving dependency management?
[Open an issue](https://github.com/fulcrum-so/pyZ3/issues) on GitHub!
