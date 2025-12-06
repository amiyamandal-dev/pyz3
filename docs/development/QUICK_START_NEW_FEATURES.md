# Quick Start Guide - New Features

Get started with the three new high-impact features in under 5 minutes!

## 🔍 Memory Leak Detection

### 1. Update Your Tests

**Before:**
```zig
test "my test" {
    py.initialize();
    defer py.finalize();
    // test code...
}
```

**After:**
```zig
test "my test" {
    var fixture = py.testing.TestFixture.init();
    defer fixture.deinit(); // 🎯 Automatic leak detection!

    fixture.initPython();
    const alloc = fixture.allocator();
    // test code...
}
```

### 2. Run Tests

```bash
poetry run pytest -v
```

If there are leaks, you'll see:
```
======================================================================
MEMORY LEAK DETECTED in 'example/leak_detection.zig::my_test'
======================================================================
The test allocator detected unreleased memory.
Please ensure all allocations are properly freed.
======================================================================
```

---

## ⚡ Hot Reload / Watch Mode

### 1. Start Watch Mode

**Option A: Basic watch (rebuild only)**
```bash
pyz3 watch --optimize Debug
```

**Option B: Watch + run Zig tests**
```bash
pyz3 watch --optimize Debug --test
```

**Option C: Watch + run pytest (recommended)**
```bash
pyz3 watch --pytest -v
```

### 2. Edit Your Code

Save any `.zig` file → Automatic rebuild! 🔄

Output:
```
🚀 Ziggy pyZ3 Watch Mode
   Optimize: Debug
   Test mode: False

👀 Watching 15 files for changes...
   Press Ctrl+C to stop

🔄 Changes detected in 1 file(s):
   - example/hello.zig

🔨 Rebuilding...
✅ Build completed in 2.34s
```

---

## 🚀 Async/Await Support

### 1. Create an Async-Friendly Function

```zig
const py = @import("pyz3");

pub fn call_async(args: struct { coro: py.PyObject }) !py.PyObject {
    // Check if it's a coroutine
    if (!py.PyCoroutine.check(args.coro)) {
        return py.TypeError(root).raise("Expected a coroutine");
    }

    // Interact with it
    const coro = py.PyCoroutine{ .obj = args.coro };
    return try coro.send(null);
}
```

### 2. Use from Python

```python
import asyncio
from my_extension import call_async

async def main():
    async def my_coro():
        await asyncio.sleep(0.1)
        return 42

    result = call_async(my_coro())
    print(result)  # 42

asyncio.run(main())
```

### 3. Create Awaitable Zig Objects

```zig
pub const AsyncTask = py.class(struct {
    const Self = @This();
    result: ?py.PyObject = null,
    done: bool = false,

    pub fn __await__(self: *const Self) !py.PyIter {
        // Make it awaitable!
        if (self.done and self.result) |res| {
            res.incref();
            const tuple = try py.tuple(root, .{res});
            return try tuple.iter();
        }
        return py.RuntimeError(root).raise("Not ready");
    }
});
```

```python
# Use from Python
task = AsyncTask()
task.set_result(123)
result = await task  # Works!
```

---

## 🎯 Complete Workflow Example

### Terminal 1: Start Watch Mode

```bash
cd /path/to/your/project
pyz3 watch --pytest -v
```

### Terminal 2: Edit Code

```zig
// example/my_feature.zig
const py = @import("pyz3");

pub fn greet(args: struct { name: []const u8 }) !py.PyString {
    return py.PyString.createFmt("Hello, {s}!", .{args.name});
}

test "greeting works" {
    var fixture = py.testing.TestFixture.init();
    defer fixture.deinit(); // Leak detection!

    fixture.initPython();

    const result = try greet(.{ .name = "World" });
    defer result.obj.decref();

    const text = try result.asSlice();
    try std.testing.expectEqualStrings("Hello, World!", text);
}

comptime {
    py.rootmodule(@This());
}
```

Save → Automatic rebuild + test! ✨

### Terminal 3: Use in Python

```python
from my_extension import greet
import asyncio

# Regular usage
print(greet("Alice"))  # Hello, Alice!

# Async context (if needed)
async def main():
    result = await some_async_operation()
    greeting = greet(result)
    print(greeting)

asyncio.run(main())
```

---

## 🐛 Troubleshooting

### Memory Leaks Not Detected

❌ **Wrong:**
```zig
test "my test" {
    py.initialize();
    defer py.finalize();
    // No leak detection!
}
```

✅ **Correct:**
```zig
test "my test" {
    var fixture = py.testing.TestFixture.init();
    defer fixture.deinit();
    fixture.initPython();
    // Leak detection enabled!
}
```

### Watch Mode Not Working

Make sure you're in a directory with `pyproject.toml`:
```bash
ls pyproject.toml  # Should exist
pyz3 watch --pytest
```

### Async Errors

Always check the type before using:
```zig
if (!py.PyCoroutine.check(obj)) {
    return py.TypeError(root).raise("Expected coroutine");
}
```

---

## 📚 Learn More

- **Full Documentation**: `docs/guide/new_features.md`
- **Examples**: `example/leak_detection.zig`, `example/async_await.zig`
- **Tests**: `test/test_new_features.py`
- **Summary**: `NEW_FEATURES_SUMMARY.md`

---

## ⚡ Cheat Sheet

```bash
# Memory Leak Detection
pytest -v  # Automatically detects leaks in Zig tests

# Watch Mode
pyz3 watch --optimize Debug          # Basic
pyz3 watch --test                    # With Zig tests
pyz3 watch --pytest -v              # With pytest (best)

# Check Available Commands
pyz3 --help
pyz3 watch --help
```

```zig
// Memory Leak Detection
var fixture = py.testing.TestFixture.init();
defer fixture.deinit();
const alloc = fixture.allocator();

// Async/Await
if (py.PyCoroutine.check(obj)) {
    const coro = py.PyCoroutine{ .obj = obj };
    const result = try coro.send(null);
}

// Create Awaitable
pub fn __await__(self: *const Self) !py.PyIter { ... }
```

---

**That's it! You're ready to use all three features!** 🎉

Start with watch mode (`pyz3 watch --pytest`) for the best development experience.
