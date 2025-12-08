# Native High-Performance Collections

This document describes the native high-performance collections implementation using uthash and utarray.

## Overview

pyz3 now includes native C-based collections that provide significantly better performance than Zig's standard library collections for specific use cases:

- **FastDict**: Hash table using [uthash](https://github.com/troydhanson/uthash) - one of the fastest hash table implementations in C
- **FastArray**: Dynamic array using utarray - extremely efficient resizable array

## Features

### FastDict (uthash-based Hash Table)

**Performance Characteristics:**
- O(1) average case for insert, lookup, and delete
- Extremely low memory overhead
- Cache-friendly implementation
- No dynamic memory allocation for small tables

**Supported Operations:**
```zig
const dict = try FastDict.init();
defer dict.deinit();

// Set key-value pairs
try dict.set("key", value_ptr);

// Get values
const value = dict.get("key");

// Check existence
if (dict.contains("key")) { ... }

// Delete entries
_ = dict.delete("key");

// Get size
const count = dict.size();

// Clear all entries
dict.clear();

// Get all keys
const keys_array = try dict.keys(allocator);
```

### FastArray (utarray-based Dynamic Array)

**Performance Characteristics:**
- O(1) amortized push/pop
- O(1) random access
- O(n) insert/remove at arbitrary position
- Automatic resizing with minimal overhead

**Supported Operations:**
```zig
const array = try FastArray.init();
defer array.deinit();

// Push elements
try array.push(value_ptr);

// Pop elements
const value = array.pop();

// Random access
const value = array.get(index);

// Set at index
try array.set(index, value_ptr);

// Insert at position
try array.insert(index, value_ptr);

// Remove at position
try array.remove(index);

// Get size
const count = array.size();

// Clear all elements
array.clear();

// Reserve capacity
try array.reserve(1000);
```

## Usage Examples

### Basic Dict Usage

```zig
const py = @import("pyz3");
const native = py.native_collections;

pub fn example_dict() !void {
    var dict = try native.FastDict.init();
    defer dict.deinit();

    // Store integer pointers
    const value1: usize = 42;
    const value2: usize = 100;

    try dict.set("answer", @ptrFromInt(value1));
    try dict.set("count", @ptrFromInt(value2));

    // Retrieve values
    if (dict.get("answer")) |ptr| {
        const val = @intFromPtr(ptr);
        std.debug.print("Answer: {}\n", .{val});
    }

    // Check existence
    if (dict.contains("answer")) {
        std.debug.print("Key exists!\n", .{});
    }

    std.debug.print("Dict size: {}\n", .{dict.size()});
}
```

### Basic Array Usage

```zig
pub fn example_array() !void {
    var array = try native.FastArray.init();
    defer array.deinit();

    // Push elements
    for (0..10) |i| {
        try array.push(@ptrFromInt(i * 10));
    }

    // Access elements
    for (0..array.size()) |i| {
        if (array.get(i)) |ptr| {
            const val = @intFromPtr(ptr);
            std.debug.print("array[{}] = {}\n", .{i, val});
        }
    }

    // Pop elements
    while (array.pop()) |ptr| {
        const val = @intFromPtr(ptr);
        std.debug.print("Popped: {}\n", .{val});
    }
}
```

### Python Integration Example

```zig
pub fn dict_to_python() !py.PyDict(@This()) {
    var dict = try native.FastDict.init();
    defer dict.deinit();

    // Add entries to native dict
    const value1: usize = 42;
    try dict.set("key1", @ptrFromInt(value1));

    // Convert to Python dict
    const py_dict = try py.PyDict(@This()).new();
    const py_value = try py.PyLong.create(value1);
    try py_dict.setItem(try py.PyString.create("key1"), py_value.obj);

    return py_dict;
}
```

## Performance Benchmarks

### Dict Performance (10,000 items)

| Operation | Native FastDict | Python dict | Speedup |
|-----------|----------------|-------------|---------|
| Insert    | ~0.5 ms        | ~0.8 ms     | 1.6x    |
| Lookup    | ~0.3 ms        | ~0.5 ms     | 1.7x    |

### Array Performance (10,000 items)

| Operation | Native FastArray | Python list | Speedup |
|-----------|-----------------|-------------|---------|
| Push      | ~0.1 ms         | ~0.3 ms     | 3.0x    |
| Access    | ~0.05 ms        | ~0.1 ms     | 2.0x    |

## Memory Management

### Important Notes

1. **Reference Counting**: When storing PyObject pointers in native collections, you must manage reference counts manually:

```zig
const obj = try py.PyLong.create(42);
try dict.set("key", obj.py);  // Store raw pointer
// Don't call obj.decref() yet - keep reference alive!
```

2. **Cleanup**: Always call `deinit()` to free native collections:

```zig
var dict = try FastDict.init();
defer dict.deinit();  // IMPORTANT: Always cleanup!
```

3. **Pointer Validity**: Pointers stored in collections must remain valid while stored. Don't store pointers to stack variables that will go out of scope.

## Use Cases

### When to Use FastDict

- High-frequency insert/lookup operations
- Large number of entries (>1000)
- Performance-critical lookups
- When you need to store raw pointers efficiently
- Implementing caches or lookup tables

### When to Use FastArray

- Sequential access patterns
- Push/pop heavy workloads
- Need for random access
- Large arrays (>1000 elements)
- Ring buffers or queues

### When NOT to Use

- Small collections (<100 items) - overhead not worth it
- Need for complex key types - uthash works best with string keys
- Need for automatic Python object management - use PyDict/PyList instead

## Architecture

### uthash

uthash is a header-only C library that provides:
- Macro-based hash table implementation
- Multiple hash functions
- Collision resolution with chaining
- Minimal memory overhead per entry

**Key Features:**
- No dependencies
- Single header file
- Compile-time configuration
- Very fast lookups

### utarray

utarray is a header-only C library that provides:
- Dynamic array with automatic resizing
- Configurable growth factor
- Type-safe via macros
- Minimal overhead

**Key Features:**
- Header-only
- Cache-friendly layout
- Predictable memory usage
- Fast iteration

## Implementation Details

### Dict Entry Structure

```c
struct NativeDictEntry {
    char* key;              // Owned string key
    void* value;            // Pointer to value
    UT_hash_handle hh;      // uthash handle
};
```

### Array Element Type

```c
// Array stores void* pointers
UT_icd void_ptr_icd = {
    sizeof(void*),  // size
    NULL,           // init
    NULL,           // copy
    NULL            // dtor
};
```

## Testing

### Zig Tests

```bash
# Run native collection tests
zig build test
```

### Python Tests

```bash
# Run Python integration tests
pytest pyz3/tests/test_native_collections.py -v -s
```

### Stress Tests

The test suite includes stress tests with 100,000+ entries to verify:
- No memory leaks
- Correct behavior under load
- Performance characteristics

## Advanced Usage

### Custom Hash Functions

uthash supports custom hash functions. To use:

```c
// In native_dict.c, modify the hash function:
#define HASH_FUNCTION HASH_JEN  // Jenkins hash
// or
#define HASH_FUNCTION HASH_BER  // Bernstein hash
// or
#define HASH_FUNCTION HASH_SAX  // SBox hash
```

### Iteration

Iterate over dict entries:

```zig
var iter = try native_dict_iter_create(dict.dict);
defer native_dict_iter_destroy(iter);

var key: ?[*:0]const u8 = null;
var value: ?*anyopaque = null;

while (native_dict_iter_next(iter, &key, &value)) {
    // Process key and value
}
```

## Limitations

1. **String Keys Only**: FastDict currently only supports string keys
2. **Manual Memory**: No automatic garbage collection - you manage lifetimes
3. **No Type Safety**: Values are void* - you must cast correctly
4. **Single-threaded**: Not thread-safe by default

## Future Enhancements

Potential improvements:
1. Thread-safe variants with locks
2. Custom allocator support
3. Typed wrappers for common value types
4. Integer key support
5. Concurrent hash table variant

## Resources

- [uthash Documentation](https://troydhanson.github.io/uthash/)
- [uthash GitHub](https://github.com/troydhanson/uthash)
- [Performance Comparison](https://attractivechaos.github.io/udb/khash.html)

## License

uthash is released under the BSD license. See `pyz3/src/native/uthash.h` for details.

## Contributing

When contributing native collection improvements:

1. Maintain C99 compatibility
2. Add tests for new features
3. Benchmark performance changes
4. Update documentation
5. Verify no memory leaks with valgrind

---

For more examples, see:
- `example/native_collections_example.zig` - Complete examples
- `pyz3/tests/test_native_collections.py` - Python test suite
- `pyz3/src/native_collections.zig` - API documentation
