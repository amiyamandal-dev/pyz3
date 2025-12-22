# PySequence Mixin

**Complete Python Sequence Protocol for Zig**

The `PySequenceMixin` provides a comprehensive set of sequence operations that can be mixed into any Zig type wrapping a Python sequence object.

## Overview

The mixin implements the complete [Python Sequence Protocol](https://docs.python.org/3/c-api/sequence.html), providing:

- ✅ **Length & Indexing** - Get/set/delete items and slices
- ✅ **Search Operations** - contains, index, count
- ✅ **Concatenation** - Join and repeat sequences
- ✅ **Iteration** - Efficient iteration with `SequenceIterator`
- ✅ **Conversion** - Convert to list, tuple, or fast sequence
- ✅ **Functional Programming** - map, filter, reduce operations
- ✅ **Type Safety** - Compile-time checked conversions

## Quick Start

### Basic Usage

```zig
const py = @import("pyz3");

pub fn example() !void {
    py.initialize();
    defer py.finalize();

    const root = @This();

    // Create a list (automatically has sequence operations)
    var list = try py.PyList(root).new(3);
    defer list.obj.decref();

    // Set items
    try list.setItem(0, 10);
    try list.setItem(1, 20);
    try list.setItem(2, 30);

    // Get length
    const length = try list.len();  // 3

    // Check membership
    const has_20 = try list.contains(20);  // true
    const has_99 = try list.contains(99);  // false

    // Find index
    const idx = try list.index(20);  // 1

    // Count occurrences
    const count = try list.count(10);  // 1
}
```

## Complete API Reference

### Core Operations

#### `len() !usize`
Get the length of the sequence.
```zig
const length = try seq.len();
```

#### `isEmpty() !bool`
Check if sequence is empty.
```zig
if (try seq.isEmpty()) {
    // Handle empty sequence
}
```

#### `getItem(comptime T: type, index: isize) !T`
Get item at index with automatic type conversion.

Supports negative indices (Python-style: -1 is last element).

```zig
const first = try list.getItem(i64, 0);
const last = try list.getItem(i64, -1);
```

#### `getItemObj(index: isize) !PyObject`
Get item at index as PyObject (no conversion).

```zig
const item = try list.getItemObj(2);
defer item.decref();
```

#### `setItem(index: isize, value: anytype) !void`
Set item at index (mutable sequences only).

```zig
try list.setItem(0, 42);
try list.setItem(1, "hello");
```

#### `delItem(index: isize) !void`
Delete item at index (mutable sequences only).

```zig
try list.delItem(0);  // Delete first item
```

### Slice Operations

#### `getSlice(start: isize, end: isize) !Self`
Get a slice of the sequence.

```zig
var slice = try list.getSlice(1, 4);  // Items 1, 2, 3
defer slice.obj.decref();
```

#### `setSlice(start: isize, end: isize, values: anytype) !void`
Replace a slice with new values.

```zig
var new_values = try py.PyList(root).new(2);
try new_values.setItem(0, 100);
try new_values.setItem(1, 200);

try list.setSlice(0, 2, new_values);
```

#### `delSlice(start: isize, end: isize) !void`
Delete a slice from the sequence.

```zig
try list.delSlice(1, 3);  // Delete items 1 and 2
```

### Search & Membership

#### `contains(value: anytype) !bool`
Check if sequence contains a value.

```zig
if (try list.contains(42)) {
    std.debug.print("Found it!\n", .{});
}
```

#### `index(value: anytype) !usize`
Find the index of the first occurrence.

Raises `ValueError` if not found.

```zig
const idx = try list.index(42);
```

#### `count(value: anytype) !usize`
Count occurrences of a value.

```zig
const occurrences = try list.count(42);
```

### Concatenation & Repetition

#### `concat(other: Self) !Self`
Concatenate two sequences.

```zig
var list1 = try py.PyList(root).new(2);
try list1.setItem(0, 1);
try list1.setItem(1, 2);

var list2 = try py.PyList(root).new(2);
try list2.setItem(0, 3);
try list2.setItem(1, 4);

var combined = try list1.concat(list2);
defer combined.obj.decref();
// Result: [1, 2, 3, 4]
```

#### `concatObj(other: PyObject) !Self`
Concatenate with any sequence-like object.

```zig
var result = try list.concatObj(some_tuple.obj);
defer result.obj.decref();
```

#### `repeat(count: usize) !Self`
Repeat sequence n times.

```zig
var list = try py.PyList(root).new(2);
try list.setItem(0, 1);
try list.setItem(1, 2);

var repeated = try list.repeat(3);
defer repeated.obj.decref();
// Result: [1, 2, 1, 2, 1, 2]
```

#### `inplaceConcat(other: Self) !void`
In-place concatenation (mutable sequences only).

```zig
try list1.inplaceConcat(list2);
// list1 is now modified
```

#### `inplaceRepeat(count: usize) !void`
In-place repeat (mutable sequences only).

```zig
try list.inplaceRepeat(3);
// list is now repeated 3 times
```

### Conversion

#### `toList() !PyList`
Convert sequence to a list.

```zig
var list = try tuple.toList();
defer list.obj.decref();
```

#### `toTuple() !PyTuple`
Convert sequence to a tuple.

```zig
var tuple = try list.toTuple();
defer tuple.obj.decref();
```

#### `fast(error_msg: [:0]const u8) !PyObject`
Get fast sequence access for efficient iteration.

```zig
const fast_seq = try list.fast("not a sequence");
defer fast_seq.decref();
// Use for tight iteration loops
```

### Utility Methods

#### `first(comptime T: type) !T`
Get the first item.

```zig
const first = try list.first(i64);
```

#### `last(comptime T: type) !T`
Get the last item.

```zig
const last = try list.last(i64);
```

#### `isValidIndex(index: isize) !bool`
Check if index is valid.

```zig
if (try list.isValidIndex(-1)) {
    // Index is valid
}
```

### Iteration

#### `iterator() SequenceIterator`
Create an iterator for the sequence.

```zig
var iter = list.iterator();
while (try iter.next()) |item| {
    defer item.decref();

    const value = try py.as(root, i64, item);
    std.debug.print("Item: {}\n", .{value});
}
```

The `SequenceIterator` provides:
- `next() !?PyObject` - Get next item (returns null when done)
- `reset() void` - Reset to beginning
- `remaining() usize` - Get remaining item count

### Functional Programming

#### `map(comptime func: anytype) !PyList`
Apply a function to each element.

```zig
const doubled = try list.map(struct {
    fn double(item: PyObject) PyObject {
        const val = py.as(root, i64, item) catch return item;
        return py.create(root, val * 2) catch item;
    }
}.double);
defer doubled.obj.decref();
```

#### `filter(comptime predicate: anytype) !PyList`
Filter sequence by predicate.

```zig
const evens = try list.filter(struct {
    fn isEven(item: PyObject) !bool {
        const val = try py.as(root, i64, item);
        return val % 2 == 0;
    }
}.isEven);
defer evens.obj.decref();
```

## Adding to Custom Types

You can add the sequence mixin to any type that has an `obj: PyObject` field:

```zig
pub const MyCustomSeq = extern struct {
    obj: PyObject,

    // Include all sequence operations
    pub usingnamespace PySequenceMixin(@This());

    // Add your custom methods
    pub fn customMethod(self: @This()) void {
        // ...
    }
};
```

## Performance Tips

### 1. Use Fast Sequences for Iteration

For tight loops, use `fast()` to get optimized sequence access:

```zig
const fast_seq = try list.fast("invalid sequence");
defer fast_seq.decref();

const len = try list.len();
var i: usize = 0;
while (i < len) : (i += 1) {
    // Much faster than repeated getItem calls
    const item = ffi.PySequence_Fast_GET_ITEM(fast_seq.py, i);
    // Use item...
}
```

### 2. Reuse Iterators

```zig
var iter = list.iterator();

// First pass
while (try iter.next()) |item| {
    defer item.decref();
    // Process item
}

// Reset and iterate again
iter.reset();
while (try iter.next()) |item| {
    defer item.decref();
    // Process item again
}
```

### 3. Batch Operations

Use in-place operations for better performance:

```zig
// Slower (creates new object)
var result = try list.concat(other);

// Faster (modifies in place)
try list.inplaceConcat(other);
```

## Error Handling

All sequence operations can raise Python exceptions:

```zig
const item = list.getItem(i64, 100) catch |err| {
    if (err == PyError.PyRaised) {
        // Python exception was raised (likely IndexError)
        // Exception is already set, will propagate
    }
    return err;
};
```

Common Python exceptions:
- `IndexError` - Index out of range
- `ValueError` - Value not found (in `index()`)
- `TypeError` - Operation not supported (e.g., setItem on tuple)

## Examples

### Example 1: Processing a List

```zig
pub fn processList(list: py.PyList(root)) !void {
    const len = try list.len();
    std.debug.print("Processing {} items\n", .{len});

    // Iterate and process
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const item = try list.getItem(i64, @intCast(i));
        const doubled = item * 2;
        try list.setItem(@intCast(i), doubled);
    }

    // Verify
    if (try list.contains(20)) {
        const idx = try list.index(20);
        std.debug.print("Found 20 at index {}\n", .{idx});
    }
}
```

### Example 2: Combining Sequences

```zig
pub fn combineSequences(
    list1: py.PyList(root),
    list2: py.PyList(root)
) !py.PyList(root) {
    // Concatenate
    var combined = try list1.concat(list2);
    errdefer combined.obj.decref();

    // Filter out negatives
    const positives = try combined.filter(struct {
        fn isPositive(item: PyObject) !bool {
            const val = try py.as(root, i64, item);
            return val > 0;
        }
    }.isPositive);
    combined.obj.decref();

    return positives;
}
```

### Example 3: Sequence Statistics

```zig
pub fn sequenceStats(seq: anytype) !struct { min: i64, max: i64, sum: i64 } {
    const len = try seq.len();
    if (len == 0) return error.EmptySequence;

    var min = try seq.getItem(i64, 0);
    var max = min;
    var sum: i64 = min;

    var i: usize = 1;
    while (i < len) : (i += 1) {
        const val = try seq.getItem(i64, @intCast(i));
        if (val < min) min = val;
        if (val > max) max = val;
        sum += val;
    }

    return .{ .min = min, .max = max, .sum = sum };
}
```

## Compatibility

The PySequence mixin works with:
- ✅ `PyList` - Full support (mutable)
- ✅ `PyTuple` - Full support (immutable, no setItem/delItem)
- ✅ `PyRange` - Read-only support
- ✅ `PyByteArray` - Full support (mutable)
- ✅ Custom sequence types - Full support

## See Also

- [Python Sequence Protocol](https://docs.python.org/3/c-api/sequence.html)
- [Python Data Model - Sequences](https://docs.python.org/3/reference/datamodel.html#sequence-types)
- PyZ3 Type System Documentation
- Example Projects

## Contributing

To add new sequence operations to the mixin:

1. Add the method to `PySequenceMixin` in `pyz3/src/types/sequence.zig`
2. Add tests to the test section
3. Document the method in this file
4. Submit a pull request

---

**For questions or issues, please open a GitHub issue with the `sequence` label.**
