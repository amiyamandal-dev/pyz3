# Memory Management Guide

Comprehensive guide to memory management in pyz3, including reference counting, allocators, and leak prevention.

## Overview

pyz3 manages two types of memory:
1. **Python objects** - Managed by reference counting
2. **Native memory** - Managed by allocators

Understanding both is critical for writing safe, performant extensions.

## Python Object Memory

### Reference Counting

Python uses reference counting for memory management. Every `PyObject` has a reference count that tracks how many references exist to it.

**Key Rule:** Every object you own must be decremented exactly once.

### Reference Ownership

#### Borrowed References

A **borrowed reference** is temporary and owned by someone else:

```zig
pub fn borrowed_example(args: struct { obj: py.PyObject }) void {
    // args.obj is a borrowed reference
    // We don't own it, so we don't decref it
    _ = args.obj;
}
```

#### New References

A **new reference** is owned by you and must be decremented:

```zig
pub fn new_reference_example() !void {
    // Create returns a new reference - we own it
    const str = try py.PyString.create("hello");

    // MUST decref when done
    defer str.obj.decref();

    // Use the string...
}
```

### Manual Reference Counting

#### Incrementing References

```zig
pub fn incref_example(args: struct { obj: py.PyObject }) py.PyObject {
    // Keep the object alive beyond this function
    args.obj.incref();

    // Return it (caller now owns the reference)
    return args.obj;
}
```

#### Decrementing References

```zig
pub fn decref_example() !void {
    const obj = try py.PyString.create("temporary");

    // Decrement when done
    defer obj.obj.decref();  // IMPORTANT!

    // Use obj...
}
```

### Common Patterns

#### Pattern: Create and Return

```zig
pub fn create_and_return() !py.PyString {
    // Create new reference
    const str = try py.PyString.create("hello");

    // Return it - caller owns the reference
    return str;
}
```

Python automatically decrefs the reference when no longer needed.

#### Pattern: Create and Use Internally

```zig
pub fn create_and_use() !i64 {
    const num = try py.PyLong.create(42);
    defer num.obj.decref();  // Cleanup

    const value = try num.as(i64);
    return value * 2;
}
```

#### Pattern: Store in Container

```zig
pub fn add_to_list() !py.PyList {
    const list = try py.PyList.new();

    const item = try py.PyLong.create(42);
    try list.append(item.obj);
    // list.append() steals the reference, so we don't decref

    // Actually, we DO need to decref because append doesn't steal!
    item.obj.decref();

    return list;
}
```

**Better pattern:**
```zig
pub fn add_to_list_better() !py.PyList {
    const list = try py.PyList.new();

    {
        const item = try py.PyLong.create(42);
        defer item.obj.decref();
        try list.append(item.obj);
    }

    return list;
}
```

### Reference Counting Helpers

#### Check Reference Count

```zig
const refcount = py.refcnt(@This(), obj);
std.debug.print("Refcount: {}\n", .{refcount});
```

#### Verify Ownership

```zig
test "reference counting" {
    py.initialize();
    defer py.finalize();

    const str = try py.PyString.create("test");
    try std.testing.expectEqual(@as(i64, 1), py.refcnt(@This(), str.obj));

    str.obj.incref();
    try std.testing.expectEqual(@as(i64, 2), py.refcnt(@This(), str.obj));

    str.obj.decref();
    str.obj.decref();
}
```

## Native Memory Management

### The PyMemAllocator

pyz3 provides `py.allocator` for native memory that integrates with Python's memory system.

#### Why Use py.allocator?

1. **GIL Caching:** Optimized to avoid redundant GIL operations
2. **Memory Tracking:** Leak detection in tests
3. **Consistency:** Same allocator Python uses internally
4. **Safety:** Proper alignment handling

### Basic Usage

```zig
pub fn allocate_buffer() ![]u8 {
    // Allocate
    const buffer = try py.allocator.alloc(u8, 1024);

    // MUST free when done
    defer py.allocator.free(buffer);

    // Use buffer...
    return buffer;
}
```

### Allocation Patterns

#### Pattern: Temporary Buffer

```zig
pub fn process_data(args: struct { size: i64 }) !void {
    const size: usize = @intCast(args.size);

    // Allocate temporary buffer
    const buffer = try py.allocator.alloc(u8, size);
    defer py.allocator.free(buffer);

    // Process...
}
```

#### Pattern: Multiple Allocations

```zig
pub fn multiple_buffers() !void {
    // GIL caching optimizes these allocations!
    const buf1 = try py.allocator.alloc(u8, 1024);
    defer py.allocator.free(buf1);

    const buf2 = try py.allocator.alloc(u8, 2048);
    defer py.allocator.free(buf2);

    const buf3 = try py.allocator.alloc(u8, 512);
    defer py.allocator.free(buf3);

    // All three allocations share one GIL acquire/release cycle
}
```

#### Pattern: Conditional Allocation

```zig
pub fn conditional_alloc(args: struct { need_buffer: bool }) !void {
    var buffer: ?[]u8 = null;
    defer if (buffer) |buf| py.allocator.free(buf);

    if (args.need_buffer) {
        buffer = try py.allocator.alloc(u8, 1024);
    }

    // Use buffer if allocated...
}
```

### Alignment

The allocator respects alignment requirements:

```zig
pub fn aligned_allocation() !void {
    // Allocate with specific alignment
    const buffer = try py.allocator.alignedAlloc(u8, 64, 1024);
    defer py.allocator.free(buffer);

    // Verify alignment
    std.debug.assert(@intFromPtr(buffer.ptr) % 64 == 0);
}
```

**Note:** Current limitation is 255 bytes max alignment. For larger alignments, use platform-specific allocators.

### Reallocation

```zig
pub fn growing_buffer() !void {
    var buffer = try py.allocator.alloc(u8, 1024);
    defer py.allocator.free(buffer);

    // Grow buffer
    buffer = try py.allocator.realloc(buffer, 2048);

    // Use larger buffer...
}
```

## Memory Leaks

### Common Leak Patterns

#### ❌ Leak: Forgot to Decref

```zig
pub fn leak_example() !void {
    const str = try py.PyString.create("leaked");
    // ERROR: Forgot defer str.obj.decref();

    // str is never decremented - MEMORY LEAK
}
```

#### ✅ Fixed: Always Defer

```zig
pub fn fixed_example() !void {
    const str = try py.PyString.create("not leaked");
    defer str.obj.decref();  // Good!

    // Use str...
}
```

#### ❌ Leak: Early Return

```zig
pub fn early_return_leak(args: struct { value: i64 }) !i64 {
    const temp = try py.PyLong.create(42);
    defer temp.obj.decref();

    if (args.value < 0) {
        return error.NegativeValue;  // OK - defer still runs
    }

    const value = try temp.as(i64);
    return value;
}
```

This is actually correct! `defer` runs even on early returns.

#### ❌ Leak: Forgot to Free Allocation

```zig
pub fn allocation_leak() !void {
    const buffer = try py.allocator.alloc(u8, 1024);
    // ERROR: Forgot defer py.allocator.free(buffer);

    // buffer is never freed - MEMORY LEAK
}
```

### Leak Detection

pyz3 includes memory leak detection in tests:

```zig
test "detect leaks" {
    const info = try py.testing.detectLeaks(@This(), struct {
        pub fn testFunc() !void {
            const str = try py.PyString.create("test");
            // Forgot to decref - will be detected!
        }
    }.testFunc);

    try std.testing.expectEqual(@as(usize, 0), info.bytes_leaked);
}
```

Output when leaks detected:
```
⚠️  Memory leak detected: 48 bytes leaked
```

### Best Practices for Preventing Leaks

1. **Always use `defer` immediately after allocation**

```zig
const obj = try create();
defer obj.decref();  // Right after creation!
```

2. **Use scopes to enforce cleanup**

```zig
{
    const temp = try create();
    defer temp.decref();
    // Use temp...
}  // Guaranteed cleanup here
```

3. **Leverage RAII patterns**

```zig
const Wrapper = struct {
    obj: py.PyObject,

    pub fn init() !@This() {
        return .{ .obj = try py.PyString.create("hello") };
    }

    pub fn deinit(self: *@This()) void {
        self.obj.decref();
    }
};

pub fn use_wrapper() !void {
    var wrapper = try Wrapper.init();
    defer wrapper.deinit();

    // Use wrapper...
}
```

## Advanced Memory Management

### Weak References

Python's weak references don't prevent garbage collection:

```zig
pub fn use_weak_ref(args: struct { obj: py.PyObject }) !py.PyObject {
    const weakref_mod = try py.import("weakref");
    defer weakref_mod.decref();

    const ref_func = try weakref_mod.getAttr("ref");
    defer ref_func.decref();

    const weak = try py.call(ref_func, .{args.obj}, .{});
    // weak is now a weak reference to obj
    return weak;
}
```

### Memory Pools

For frequently allocated/deallocated objects of the same size:

```zig
const Pool = struct {
    const BLOCK_SIZE = 1024;
    blocks: std.ArrayList([]u8),

    pub fn init() !@This() {
        return .{
            .blocks = std.ArrayList([]u8).init(py.allocator),
        };
    }

    pub fn alloc(self: *@This()) ![]u8 {
        const block = try py.allocator.alloc(u8, BLOCK_SIZE);
        try self.blocks.append(block);
        return block;
    }

    pub fn deinit(self: *@This()) void {
        for (self.blocks.items) |block| {
            py.allocator.free(block);
        }
        self.blocks.deinit();
    }
};
```

### Buffer Protocol

For zero-copy memory sharing with Python:

```zig
pub fn create_buffer() !py.PyBuffer {
    const data = try py.allocator.alloc(u8, 1024);
    // Python can access this buffer without copying

    // Note: Buffer management is complex - see NumPy integration
    return py.PyBuffer.init(data);
}
```

## Memory Profiling

### Tracking Allocations

```zig
test "track allocations" {
    py.initialize();
    defer py.finalize();

    const before = getMemoryUsage();

    {
        var allocations = std.ArrayList([]u8).init(std.testing.allocator);
        defer allocations.deinit();

        // Make many allocations
        for (0..1000) |_| {
            const buf = try py.allocator.alloc(u8, 1024);
            defer py.allocator.free(buf);
        }
    }

    const after = getMemoryUsage();
    std.debug.print("Memory used: {} bytes\n", .{after - before});
}
```

### Memory Limits

Set limits to prevent runaway allocation:

```python
import resource

# Limit memory to 1GB
resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024, 1024*1024*1024))
```

## GIL and Memory

### GIL Acquisition for Allocations

The `py.allocator` automatically manages the GIL:

```zig
pub fn gil_managed_alloc() !void {
    // GIL acquired automatically
    const buf = try py.allocator.alloc(u8, 1024);
    defer py.allocator.free(buf);  // GIL acquired for free too

    // With GIL caching, nested allocations are optimized
}
```

### Manual GIL Management

Sometimes you need manual control:

```zig
pub fn manual_gil() !void {
    // Acquire GIL
    const gil = py.gil();
    defer gil.release();

    // Safe to call Python C API here

    const obj = try py.PyString.create("test");
    defer obj.obj.decref();
}
```

### Releasing GIL for I/O

```zig
pub fn io_operation() !void {
    const data_obj = try py.PyString.create("data");
    defer data_obj.decref();

    // Release GIL for blocking I/O
    {
        const nogil = py.nogil();
        defer nogil.reacquire();

        // Perform I/O without holding GIL
        // Other Python threads can run
        std.time.sleep(1_000_000_000);  // 1 second
    }

    // GIL reacquired automatically
}
```

## Performance Tips

### 1. Minimize Allocations

```zig
// Bad: Allocates every call
pub fn bad_pattern(args: struct { value: i64 }) !py.PyLong {
    return try py.PyLong.create(args.value);
}

// Good: Use small int pool
pub fn good_pattern(args: struct { value: i64 }) !py.PyLong {
    // If value is -5 to 256, uses cached object
    return try py.PyLong.create(args.value);
}
```

### 2. Batch Operations

```zig
// Bad: Many small allocations
pub fn bad_batch() !void {
    for (0..1000) |_| {
        const buf = try py.allocator.alloc(u8, 10);
        defer py.allocator.free(buf);
    }
}

// Good: One large allocation
pub fn good_batch() !void {
    const buf = try py.allocator.alloc(u8, 10000);
    defer py.allocator.free(buf);

    // Use in chunks of 10
}
```

### 3. Reuse Allocations

```zig
pub fn reuse_buffer() !void {
    var buffer = try py.allocator.alloc(u8, 1024);
    defer py.allocator.free(buffer);

    // Reuse buffer in loop
    for (0..100) |i| {
        // Clear buffer
        @memset(buffer, 0);

        // Fill with new data
        buffer[0] = @intCast(i);

        // Process...
    }
}
```

## Debugging Memory Issues

### Enable Debug Allocations

```zig
const std = @import("std");
const debug_allocator = std.heap.GeneralPurposeAllocator(.{
    .safety = true,
    .thread_safe = true,
}){};
```

### Add Assertions

```zig
pub fn checked_allocation() !void {
    const buf = try py.allocator.alloc(u8, 1024);
    defer py.allocator.free(buf);

    // Assert buffer is non-null
    std.debug.assert(buf.len == 1024);

    // Assert alignment
    std.debug.assert(@intFromPtr(buf.ptr) % @alignOf(u8) == 0);
}
```

### Track Reference Counts

```zig
pub fn track_refs() !void {
    const str = try py.PyString.create("test");
    defer str.obj.decref();

    std.debug.print("Initial refcount: {}\n", .{py.refcnt(@This(), str.obj)});

    str.obj.incref();
    std.debug.print("After incref: {}\n", .{py.refcnt(@This(), str.obj)});

    str.obj.decref();
    std.debug.print("After decref: {}\n", .{py.refcnt(@This(), str.obj)});
}
```

## See Also

- [API Reference](README.md)
- [Performance Guide](performance.md)
- [Type Conversion](type-conversion.md)
- [Testing Guide](../guide/testing.md)
