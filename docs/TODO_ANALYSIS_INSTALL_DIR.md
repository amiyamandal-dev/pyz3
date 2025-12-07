# Analysis: TODO(ngates) - Find Install Directory Properly

## Location
**Files:** `pyz3.build.zig:281`, `pyz3/src/pyz3.build.zig:281`

## Current Code
```zig
// Install the shared library within the source tree
const install = b.addInstallFileWithDir(
    lib.getEmittedBin(),
    // TODO(ngates): find this somehow?
    .{ .custom = ".." }, // Relative to project root: zig-out/../
    libraryDestRelPath(self.allocator, options) catch |err| {
        std.debug.print("\n❌ Out of memory computing library destination path\n", .{});
        std.debug.print("   Error: {}\n", .{err});
        std.process.exit(1);
    },
);
```

## Problem Statement

The code needs to install compiled Python extension modules (`.abi3.so` files) into the **source tree** rather than the standard Zig output directory (`zig-out/`).

### Why This Is Needed

1. **Python Import Mechanism**: Python imports expect modules at:
   ```
   package/subpackage/module.abi3.so
   ```

2. **Standard Zig Behavior**: By default, Zig installs artifacts to:
   ```
   zig-out/lib/module.abi3.so
   ```

3. **pyz3 Requirement**: Modules must be installed directly in the source tree:
   ```
   project_root/
   ├── package/
   │   └── module.abi3.so  ← Must be here for Python to import
   └── zig-out/
       └── lib/module.abi3.so  ← Standard Zig location
   ```

### Current Workaround

Using `.{ .custom = ".." }` works because:
- Zig build runs from `project_root/`
- Default install prefix is `zig-out/` (relative to project_root)
- To install to `project_root/package/`, we need to escape back: `zig-out/../package/`

## Why It's a TODO

The `.{ .custom = ".." }` approach is:
1. **Hacky**: Relies on implementation detail that zig-out is one level deep
2. **Fragile**: Could break if Zig changes directory structure
3. **Unclear**: Not semantically obvious what ".." represents
4. **Not Portable**: Assumes Unix-style path separators

## Zig Build System Context

### InstallDir Enum

From Zig's build system, `InstallDir` has these variants:
```zig
pub const InstallDir = union(enum) {
    prefix,           // zig-out/ (default prefix)
    lib,             // zig-out/lib/
    bin,             // zig-out/bin/
    header,          // zig-out/include/
    custom: []const u8,  // Custom path relative to prefix
};
```

### The Challenge

None of the standard `InstallDir` options allow installing **outside** the prefix directory. The `custom` variant still installs **relative to prefix**, so:
- `.{ .custom = "package" }` → `zig-out/package/` ❌
- `.{ .custom = "../package" }` → `zig-out/../package/` = `package/` ✓ (current workaround)

## Potential Solutions

### Option 1: Use `InstallDir.prefix` with Relative Subdirectory ✓ **RECOMMENDED**

Instead of `.{ .custom = ".." }`, calculate the full path:

```zig
// Calculate path relative to prefix (zig-out/)
// For module "package.subpkg.module" → "../package/subpkg/module.abi3.so"
fn calculateInstallPath(allocator: std.mem.Allocator, module_path: []const u8) ![]const u8 {
    // Count directory depth in module path
    const depth = std.mem.count(u8, module_path, "/");

    // Build "../" prefix to escape zig-out
    const escape_prefix = "../";
    const total_len = escape_prefix.len + module_path.len;

    const result = try allocator.alloc(u8, total_len);
    @memcpy(result[0..escape_prefix.len], escape_prefix);
    @memcpy(result[escape_prefix.len..], module_path);

    return result;
}

// Usage:
const install_path = try calculateInstallPath(
    self.allocator,
    libraryDestRelPath(self.allocator, options)
);
defer self.allocator.free(install_path);

const install = b.addInstallFileWithDir(
    lib.getEmittedBin(),
    .{ .custom = install_path },
    "",  // No additional subdirectory needed
);
```

**Pros:**
- Explicit and clear
- Documents the intent
- Still uses standard Zig API

**Cons:**
- Still a workaround
- Duplicates ".." logic

### Option 2: Use Absolute Path Installation

Create a custom step that copies to absolute path:

```zig
const CopyToSourceTree = struct {
    step: Step,
    source: LazyPath,
    dest_path: []const u8,

    pub fn create(b: *std.Build, source: LazyPath, dest_path: []const u8) *CopyToSourceTree {
        const self = b.allocator.create(CopyToSourceTree) catch @panic("OOM");
        self.* = .{
            .step = Step.init(.{
                .id = .custom,
                .name = "copy to source tree",
                .owner = b,
                .makeFn = make,
            }),
            .source = source,
            .dest_path = dest_path,
        };
        return self;
    }

    fn make(step: *Step, _: std.Progress.Node) !void {
        const self: *CopyToSourceTree = @fieldParentPtr("step", step);
        const b = step.owner;

        const source_path = self.source.getPath2(b, step);

        // Copy file to absolute destination
        const cwd = std.fs.cwd();
        try cwd.makePath(std.fs.path.dirname(self.dest_path) orelse ".");
        try cwd.copyFile(source_path, cwd, self.dest_path, .{});
    }
};

// Usage:
const abs_dest = try std.fmt.allocPrint(
    self.allocator,
    "{s}",
    .{libraryDestRelPath(self.allocator, options)}
);
const copy_step = CopyToSourceTree.create(b, lib.getEmittedBin(), abs_dest);
b.getInstallStep().dependOn(&copy_step.step);
```

**Pros:**
- Clean, explicit absolute path handling
- No path manipulation hacks
- Clear intent

**Cons:**
- More code
- Bypasses Zig's install system
- Need to handle cross-platform paths

### Option 3: Keep Current Implementation with Better Documentation ✓ **SIMPLEST**

Add clear comments explaining why ".." is used:

```zig
// Install the shared library within the source tree (not zig-out/).
// Python imports expect modules at: package/subpackage/module.abi3.so
// We use .{ .custom = ".." } to install relative to project root instead of zig-out/.
// This maps: zig-out/../package/module.so → package/module.so
const install = b.addInstallFileWithDir(
    lib.getEmittedBin(),
    .{ .custom = ".." }, // Escape zig-out/ to install in project root
    libraryDestRelPath(self.allocator, options) catch |err| {
        std.debug.print("\n❌ Out of memory computing library destination path\n", .{});
        std.debug.print("   Error: {}\n", .{err});
        std.process.exit(1);
    },
);
```

**Pros:**
- Minimal change
- Already works
- Clear documentation

**Cons:**
- Still a workaround
- Relies on implementation detail

## Recommended Solution

**Option 3** (Better Documentation) is recommended because:

1. **It works**: The current code is functional and tested
2. **Minimal risk**: No behavior changes
3. **Clear intent**: Good comments make it understandable
4. **Stable API**: Zig's `InstallDir` API is unlikely to change in breaking ways

The "TODO" can be resolved by:
1. Adding comprehensive comments explaining the rationale
2. Changing comment from "TODO: find this somehow?" to clear explanation
3. Optionally wrapping in a helper function

## Implementation

### Recommended Change:

```zig
// Helper function to document the install directory choice
fn getSourceTreeInstallDir() std.Build.InstallDir {
    // We install Python extension modules directly into the source tree
    // (e.g., package/module.abi3.so) rather than zig-out/, because Python's
    // import mechanism expects modules to be co-located with Python source files.
    //
    // Using .{ .custom = ".." } installs relative to zig-out/, which resolves to:
    // zig-out/../package/module.abi3.so → package/module.abi3.so
    //
    // This is the intended behavior and aligns with how Python build systems work.
    return .{ .custom = ".." };
}

// Usage:
const install = b.addInstallFileWithDir(
    lib.getEmittedBin(),
    getSourceTreeInstallDir(),
    libraryDestRelPath(self.allocator, options) catch |err| {
        std.debug.print("\n❌ Out of memory computing library destination path\n", .{});
        std.debug.print("   Error: {}\n", .{err});
        std.process.exit(1);
    },
);
```

## Alternative: If Zig Adds New API

If future Zig versions add a way to install outside the prefix, we could use:
```zig
// Hypothetical future API:
const install = b.addInstallFileToPath(
    lib.getEmittedBin(),
    b.pathFromRoot(libraryDestRelPath(...)),
);
```

But this doesn't exist in current Zig (as of 0.15).

## Conclusion

**Status**: ✅ Can be resolved with documentation

**Action**: Replace TODO with clear explanation or helper function

**Priority**: Low - current code works correctly

**Breaking**: No - purely a code clarity improvement

## References

- [Zig Build System Documentation](https://ziglang.org/learn/build-system/)
- [Zig Build API (zig.guide)](https://zig.guide/build-system/zig-build/)
- [Understanding build.zig (DEV Community)](https://dev.to/hexshift/understanding-buildzig-a-practical-introduction-to-zigs-build-system-6gh)
