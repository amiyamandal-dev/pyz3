# zigimport Advanced Features Guide

Complete guide to all advanced features in zigimport.

## Table of Contents

1. [Dependency Tracking](#dependency-tracking)
2. [Custom build.zig Support](#custom-buildzig-support)
3. [Parallel Compilation](#parallel-compilation)
4. [Watch Mode](#watch-mode)
5. [Remote Caching](#remote-caching)
6. [PyPI Distribution](#pypi-distribution)
7. [Configuration Reference](#configuration-reference)

---

## 1. Dependency Tracking

zigimport automatically tracks imported `.zig` files and recompiles when any dependency changes.

### How It Works

```zig
// main.zig
const utils = @import("utils.zig");  // Tracked!
const math = @import("math.zig");    // Tracked!

pub fn process(args: struct { x: i64 }) i64 {
    return utils.transform(math.calculate(args.x));
}
```

**Automatic tracking:**
- Parses `@import("...")` statements
- Recursively tracks transitive dependencies
- Recompiles when ANY dependency changes
- Hashes all files for cache invalidation

### Example

```
Project structure:
my_module.zig
├── imports utils.zig
│   └── imports helpers.zig
└── imports math.zig

When you edit helpers.zig → my_module recompiles automatically!
```

### Configuration

```python
import os

# Enable dependency tracking (default: enabled)
os.environ["ZIGIMPORT_TRACK_DEPS"] = "1"

# Disable if you want faster imports (only tracks main file)
os.environ["ZIGIMPORT_TRACK_DEPS"] = "0"

import pyz3.zigimport
```

### Benefits

- ✅ Always uses latest code
- ✅ No stale modules
- ✅ Works with complex dependency graphs
- ✅ Automatic recursive tracking

---

## 2. Custom build.zig Support

Use your own `build.zig` for complex build configurations.

### Structure

```
my_complex_module/
├── build.zig          # Custom build configuration
├── src/
│   ├── main.zig
│   ├── lib.zig
│   └── helpers.zig
└── include/
    └── myheader.h
```

### Example build.zig

```zig
const std = @import("std");
const py = @import("pyz3");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addSharedLibrary(.{
        .name = "my_complex_module",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add C includes
    lib.addIncludePath(b.path("include"));

    // Link C library
    lib.linkSystemLibrary("sqlite3");

    // Add custom compile flags
    lib.addCSourceFile(.{
        .file = b.path("src/native.c"),
        .flags = &.{"-O3", "-fPIC"},
    });

    b.installArtifact(lib);
}
```

### Usage

```python
import pyz3.zigimport
import my_complex_module  # Uses custom build.zig!

my_complex_module.advanced_function()
```

### zigimport Detection

zigimport searches for:
1. `module_name.zig` (simple module)
2. `module_name/build.zig` (custom build)

If `build.zig` exists, it takes precedence.

### Benefits

- ✅ Full control over build process
- ✅ Link external libraries
- ✅ Custom compilation flags
- ✅ Multi-file projects
- ✅ C/C++ integration

---

## 3. Parallel Compilation

Compile multiple modules concurrently for faster builds.

### Enable Parallel Compilation

```python
import os

# Enable parallel builds (default: enabled)
os.environ["ZIGIMPORT_PARALLEL"] = "1"

# Set max worker threads (default: 4)
os.environ["ZIGIMPORT_MAX_WORKERS"] = "8"

import pyz3.zigimport

# These compile in parallel!
import module1
import module2
import module3
import module4
```

### How It Works

```
Traditional (Sequential):
  module1 → module2 → module3 → module4
  Total: 12 seconds

Parallel (4 workers):
  module1 ┐
  module2 ├→ All done!
  module3 │
  module4 ┘
  Total: 3 seconds
```

### Use Cases

**Large Projects:**
```python
# Compile 10 modules in parallel
import os
os.environ["ZIGIMPORT_MAX_WORKERS"] = "10"

import pyz3.zigimport

# All compile concurrently
from my_package import (
    module1, module2, module3, module4, module5,
    module6, module7, module8, module9, module10
)
```

**CI/CD:**
```bash
# Use all CPU cores
export ZIGIMPORT_MAX_WORKERS=$(nproc)
export ZIGIMPORT_PARALLEL=1

python -m pytest  # Tests run with parallel compilation
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ZIGIMPORT_PARALLEL` | `1` | Enable parallel compilation |
| `ZIGIMPORT_MAX_WORKERS` | `4` | Max concurrent compilations |

### Benefits

- ⚡ 2-4x faster for multiple modules
- 🚀 Utilizes multi-core CPUs
- 📦 Perfect for CI/CD pipelines
- 🔧 Configurable thread pool

---

## 4. Watch Mode

Automatically reload modules when source files change.

### Enable Watch Mode

```python
import os
os.environ["ZIGIMPORT_WATCH"] = "1"
os.environ["ZIGIMPORT_WATCH_INTERVAL"] = "1.0"  # Check every 1 second

import pyz3.zigimport
import my_module

# Edit my_module.zig → automatically reloads!
```

### Interactive Development

```python
import os
os.environ["ZIGIMPORT_WATCH"] = "1"
os.environ["ZIGIMPORT_VERBOSE"] = "1"

import pyz3.zigimport
import my_module

# Start Python REPL or Jupyter
while True:
    result = my_module.process(data)
    print(result)

    # Edit my_module.zig in your editor
    # Watch mode detects changes and reloads automatically!
    # Output: [zigimport] Change detected: my_module.zig
    #         [zigimport] Reloaded: my_module
```

### Jupyter Notebook Example

```python
# Cell 1: Enable watch mode
import os
os.environ["ZIGIMPORT_WATCH"] = "1"

%load_ext pyz3.import_hook
import my_module

# Cell 2: Use the module
result = my_module.compute(42)
print(result)

# Edit my_module.zig...
# Re-run Cell 2 → sees updated code automatically!
```

### Watch All Dependencies

Watch mode tracks:
- Main `.zig` file
- All imported dependencies
- Transitive imports

```
my_module.zig  → Watched
├── utils.zig  → Watched
└── math.zig   → Watched
    └── constants.zig → Watched
```

Edit ANY file → automatic reload!

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ZIGIMPORT_WATCH` | `0` | Enable watch mode |
| `ZIGIMPORT_WATCH_INTERVAL` | `1.0` | Check interval (seconds) |

### Programmatic Control

```python
import pyz3.zigimport as zi

# Enable watch mode
zi.enable_watch_mode()

# Disable watch mode
zi.disable_watch_mode()
```

### Benefits

- 🔄 Auto-reload on file changes
- 🎯 Watches all dependencies
- 📝 Perfect for development
- 🚀 No manual recompilation

---

## 5. Remote Caching

Share compiled modules across machines for faster builds.

### Setup

```python
import os

# Point to shared cache directory (NFS, S3, etc.)
os.environ["ZIGIMPORT_REMOTE_CACHE"] = "/shared/zigimport-cache"

# Enable cache uploads
os.environ["ZIGIMPORT_CACHE_UPLOAD"] = "1"

import pyz3.zigimport
import my_module  # Checks remote cache first!
```

### How It Works

```
Machine A:
  1. Compiles my_module.zig → my_module.so
  2. Uploads to /shared/zigimport-cache/abc123.so

Machine B:
  1. Imports my_module
  2. Checks remote cache
  3. Downloads abc123.so → instant!
  4. No compilation needed
```

### Use Cases

**Team Development:**
```bash
# Shared NFS mount
export ZIGIMPORT_REMOTE_CACHE=/mnt/team-cache
export ZIGIMPORT_CACHE_UPLOAD=1

# First developer compiles → uploads
python -c "import pyz3.zigimport; import my_module"

# Other developers → download from cache (instant!)
```

**CI/CD Pipeline:**
```yaml
# .github/workflows/test.yml
- name: Setup cache
  run: |
    export ZIGIMPORT_REMOTE_CACHE=${{ github.workspace }}/cache
    export ZIGIMPORT_CACHE_UPLOAD=1

- name: Run tests
  run: pytest  # Uses cached compilations
```

**Docker Builds:**
```dockerfile
FROM python:3.11

# Mount cache volume
VOLUME /zigimport-cache

ENV ZIGIMPORT_REMOTE_CACHE=/zigimport-cache
ENV ZIGIMPORT_CACHE_UPLOAD=1

RUN pip install pyz3
```

### Cache Key

Modules are cached by hash of:
- Source code content
- All dependencies
- Build configuration (optimize level)
- Python version

**Same hash = cache hit!**

### Backends

Remote cache works with:
- ✅ NFS mounts
- ✅ Shared directories
- ✅ S3/GCS (with FUSE mount)
- ✅ Any network filesystem

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ZIGIMPORT_REMOTE_CACHE` | `` | Path to remote cache dir |
| `ZIGIMPORT_CACHE_UPLOAD` | `0` | Upload compiled modules |

### Benefits

- 🚀 Instant imports on cache hit
- 👥 Share builds across team
- 💾 Reduce CI/CD build times
- 🌍 Works with any shared storage

---

## 6. PyPI Distribution

Pre-compile modules during wheel building for faster installation.

### Setup in setup.py

```python
from setuptools import setup
from setuptools.command.build_ext import build_ext
import pyz3.zigimport as zi

class BuildZigModules(build_ext):
    def run(self):
        # Pre-compile all .zig modules
        import os
        os.environ["ZIGIMPORT_OPTIMIZE"] = "ReleaseFast"

        # Import to trigger compilation
        import my_package.module1
        import my_package.module2

        # Copy compiled .so files to package
        # (implementation depends on your structure)

        super().run()

setup(
    name="my-package",
    version="1.0.0",
    packages=["my_package"],
    cmdclass={
        "build_ext": BuildZigModules,
    },
)
```

### Using in pyproject.toml

```toml
[build-system]
requires = ["setuptools", "pyz3"]
build-backend = "setuptools.build_meta"

[project]
name = "my-package"
version = "1.0.0"

[tool.zigimport.precompile]
optimize = "ReleaseFast"
modules = [
    "my_package.module1",
    "my_package.module2",
]
```

### Benefits

- 📦 Users don't need Zig installed
- ⚡ Faster installation (no compilation)
- 🎯 Pre-optimized binaries
- 🌍 Cross-platform wheels

### Example: Creating Wheels

```bash
# Build wheel with pre-compiled modules
export ZIGIMPORT_OPTIMIZE=ReleaseFast
python -m build --wheel

# Result: my_package-1.0.0-py3-none-any.whl
# Contains pre-compiled .so files!
```

### Distribution Strategy

**For Library Authors:**
1. Pre-compile during `python -m build`
2. Include `.so` files in wheel
3. Ship platform-specific wheels

**For End Users:**
```bash
pip install my-package  # Just works, no Zig needed!
```

---

## Configuration Reference

### Complete Environment Variables

| Variable | Default | Values | Description |
|----------|---------|--------|-------------|
| `ZIGIMPORT_OPTIMIZE` | `Debug` | Debug, ReleaseSafe, ReleaseFast, ReleaseSmall | Optimization level |
| `ZIGIMPORT_VERBOSE` | `0` | 0, 1 | Enable verbose logging |
| `ZIGIMPORT_FORCE_REBUILD` | `0` | 0, 1 | Always recompile |
| `ZIGIMPORT_BUILD_DIR` | `~/.zigimport` | path | Cache directory |
| `ZIGIMPORT_PARALLEL` | `1` | 0, 1 | Enable parallel compilation |
| `ZIGIMPORT_MAX_WORKERS` | `4` | 1-32 | Max concurrent builds |
| `ZIGIMPORT_WATCH` | `0` | 0, 1 | Enable watch mode |
| `ZIGIMPORT_WATCH_INTERVAL` | `1.0` | seconds | File check interval |
| `ZIGIMPORT_REMOTE_CACHE` | `` | path | Remote cache directory |
| `ZIGIMPORT_CACHE_UPLOAD` | `0` | 0, 1 | Upload to remote cache |
| `ZIGIMPORT_TRACK_DEPS` | `1` | 0, 1 | Track dependencies |

### Example Configurations

**Development:**
```bash
export ZIGIMPORT_OPTIMIZE=Debug
export ZIGIMPORT_VERBOSE=1
export ZIGIMPORT_WATCH=1
export ZIGIMPORT_TRACK_DEPS=1
```

**Production:**
```bash
export ZIGIMPORT_OPTIMIZE=ReleaseFast
export ZIGIMPORT_VERBOSE=0
export ZIGIMPORT_PARALLEL=1
export ZIGIMPORT_MAX_WORKERS=8
export ZIGIMPORT_REMOTE_CACHE=/shared/cache
```

**CI/CD:**
```bash
export ZIGIMPORT_OPTIMIZE=ReleaseFast
export ZIGIMPORT_PARALLEL=1
export ZIGIMPORT_MAX_WORKERS=$(nproc)
export ZIGIMPORT_REMOTE_CACHE=$CI_CACHE_DIR
export ZIGIMPORT_CACHE_UPLOAD=1
```

---

## Complete Example

Putting it all together:

```python
import os

# Configure for optimal development
os.environ.update({
    "ZIGIMPORT_OPTIMIZE": "Debug",
    "ZIGIMPORT_VERBOSE": "1",
    "ZIGIMPORT_WATCH": "1",
    "ZIGIMPORT_WATCH_INTERVAL": "0.5",
    "ZIGIMPORT_PARALLEL": "1",
    "ZIGIMPORT_MAX_WORKERS": "4",
    "ZIGIMPORT_TRACK_DEPS": "1",
    "ZIGIMPORT_REMOTE_CACHE": "/shared/team-cache",
    "ZIGIMPORT_CACHE_UPLOAD": "1",
})

import pyz3.zigimport

# Import modules - all advanced features active!
import my_module1  # Checks remote cache, compiles if needed
import my_module2  # Compiles in parallel with module1
import my_module3  # Tracks dependencies, watches for changes

# Use modules
result = my_module1.process(data)

# Edit any .zig file → automatic reload!
# All dependencies tracked
# Uploads to shared cache for teammates
```

---

## Performance Comparison

| Feature | Disabled | Enabled | Speedup |
|---------|----------|---------|---------|
| Dependency tracking | 1 file checked | All deps checked | Correctness ++ |
| Parallel compilation | 10s sequential | 3s parallel | 3.3x |
| Watch mode | Manual reload | Auto reload | ∞x better DX |
| Remote cache | Always compile | Download | 100x on cache hit |

---

## Summary

zigimport advanced features provide:

- ✅ **Dependency Tracking** - Always use latest code
- ✅ **Custom build.zig** - Full build control
- ✅ **Parallel Compilation** - 2-4x faster builds
- ✅ **Watch Mode** - Auto-reload on changes
- ✅ **Remote Caching** - Share builds across machines
- ✅ **PyPI Distribution** - Pre-compiled wheels

All features work together seamlessly for the ultimate Zig-Python development experience!
