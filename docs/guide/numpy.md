# NumPy Integration

pyZ3 provides seamless integration with NumPy arrays, enabling zero-copy data access and efficient array operations directly from Zig code.

## Overview

The NumPy integration allows you to:

- **Create NumPy arrays** from Zig slices
- **Access array data** with zero-copy performance
- **Modify arrays in-place** using Zig's type safety
- **Call NumPy methods** directly from Zig
- **Work with multiple dimensions** and shapes

## Quick Start

```zig
const py = @import("pyz3");

pub fn process_array(args: struct { arr: py.PyArray(@This()) }) !py.PyArray(@This()) {
    // Zero-copy access to array data
    const data = try args.arr.asSliceMut(f64);

    // Modify in-place using Zig
    for (data) |*val| {
        val.* *= 2.0;
    }

    return args.arr;
}

comptime {
    py.rootmodule(@This());
}
```

```python
import numpy as np
import mymodule

arr = np.array([1.0, 2.0, 3.0])
result = mymodule.process_array(arr)
print(result)  # [2.0, 4.0, 6.0]
```

## Type System

### DType Enum

pyZ3 provides a `DType` enum that maps Zig types to NumPy dtypes:

```zig
pub const DType = enum(c_int) {
    bool = 0,
    int8 = 1,
    uint8 = 2,
    int16 = 3,
    uint16 = 4,
    int32 = 5,
    uint32 = 6,
    int64 = 7,
    uint64 = 8,
    float32 = 11,
    float64 = 12,
    complex64 = 14,
    complex128 = 15,
};
```

The dtype system provides compile-time type safety:

```zig
// Get dtype from Zig type at compile time
const dtype = py.DType.fromType(f64);  // Returns DType.float64
```

### PyArray Type

The `PyArray` type wraps NumPy arrays with type-safe Zig operations:

```zig
pub fn PyArray(comptime root: type) type
```

Each function must specify its root module when using `PyArray`:

```zig
const root = @This();

pub fn my_function(args: struct { arr: py.PyArray(root) }) !void {
    // ... work with array
}
```

## Creating Arrays

### From Zig Slices

Create NumPy arrays directly from Zig slices (copies data):

```zig
pub fn create_array() !py.PyArray(@This()) {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    return try py.PyArray(@This()).fromSlice(f64, &data);
}
```

### 2D Arrays

Create multi-dimensional arrays:

```zig
pub fn create_matrix() !py.PyArray(@This()) {
    const data = [_]f64{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    return try py.PyArray(@This()).fromSlice2D(f64, &data, 2, 3);
}
```

### Array Constructors

Use NumPy-style constructors:

```zig
// Create array of zeros
pub fn make_zeros() !py.PyArray(@This()) {
    return try py.PyArray(@This()).zeros(f64, &[_]usize{ 10, 10 });
}

// Create array of ones
pub fn make_ones() !py.PyArray(@This()) {
    return try py.PyArray(@This()).ones(i32, &[_]usize{100});
}

// Create array filled with value
pub fn make_constant() !py.PyArray(@This()) {
    return try py.PyArray(@This()).full(f64, &[_]usize{5, 5}, 3.14);
}
```

## Zero-Copy Access

### Read-Only Access

Access array data without copying:

```zig
pub fn sum_array(args: struct { arr: py.PyArray(@This()) }) !f64 {
    const data = try args.arr.asSlice(f64);

    var sum: f64 = 0.0;
    for (data) |val| {
        sum += val;
    }

    return sum;
}
```

### Mutable Access

Modify array data in-place:

```zig
pub fn double_values(args: struct { arr: py.PyArray(@This()) }) !py.PyArray(@This()) {
    const data = try args.arr.asSliceMut(f64);

    for (data) |*val| {
        val.* *= 2.0;
    }

    return args.arr;
}
```

**Important**: Mutable access returns a writable view of the original array. Changes are reflected in the Python NumPy array.

## Array Metadata

### Shape and Dimensions

```zig
pub fn get_info(args: struct { arr: py.PyArray(@This()) }) !void {
    // Get shape as tuple
    const shape = try args.arr.shape();
    defer shape.decref();

    // Get number of dimensions
    const ndim = try args.arr.ndim();

    // Get total number of elements
    const size = try args.arr.size();

    // Get data type
    const dtype = try args.arr.dtype();
    defer dtype.decref();
}
```

## Array Operations

### Reshaping

```zig
pub fn reshape_matrix(args: struct { arr: py.PyArray(@This()) }) !py.PyArray(@This()) {
    // Reshape to 5x2
    return try args.arr.reshape(&[_]usize{ 5, 2 });
}
```

### Transposing

```zig
pub fn transpose_matrix(args: struct { arr: py.PyArray(@This()) }) !py.PyArray(@This()) {
    return try args.arr.transpose();
}
```

### Flattening

```zig
pub fn flatten_array(args: struct { arr: py.PyArray(@This()) }) !py.PyArray(@This()) {
    return try args.arr.flatten();
}
```

## Reductions

Call NumPy reduction methods directly:

```zig
pub fn array_statistics(args: struct { arr: py.PyArray(@This()) }) !struct {
    min: f64,
    max: f64,
    mean: f64,
    sum: f64
} {
    return .{
        .min = try args.arr.min(f64),
        .max = try args.arr.max(f64),
        .mean = try args.arr.mean(f64),
        .sum = try args.arr.sum(f64),
    };
}
```

## Multi-Array Operations

Combine multiple arrays efficiently:

```zig
pub fn element_wise_add(args: struct {
    a: py.PyArray(@This()),
    b: py.PyArray(@This())
}) !py.PyArray(@This()) {
    // Verify same size
    const size_a = try args.a.size();
    const size_b = try args.b.size();

    if (size_a != size_b) {
        return py.ValueError(@This()).raise("Arrays must have same size");
    }

    // Get data from both arrays
    const data_a = try args.a.asSlice(f64);
    const data_b = try args.b.asSlice(f64);

    // Create result array
    const result = try py.PyArray(@This()).zeros(f64, &[_]usize{size_a});
    const result_data = try result.asSliceMut(f64);

    // Perform operation
    for (data_a, data_b, result_data) |a, b, *r| {
        r.* = a + b;
    }

    return result;
}
```

## Classes with Arrays

Store and manipulate arrays in Zig classes:

```zig
pub const DataProcessor = py.class(struct {
    pub const __doc__ = "Process NumPy arrays";
    const Self = @This();

    data: py.PyArray(@This()),

    pub fn __init__(self: *Self, args: struct { arr: py.PyArray(@This()) }) !void {
        self.* = .{ .data = args.arr };
    }

    pub fn get_mean(self: *const Self) !f64 {
        return try self.data.mean(f64);
    }

    pub fn normalize(self: *const Self) !py.PyArray(@This()) {
        const mean_val = try self.data.mean(f64);
        const data = try self.data.asSliceMut(f64);

        for (data) |*val| {
            val.* -= mean_val;
        }

        return self.data;
    }
});
```

## Performance Tips

### Zero-Copy is Key

Use `asSlice()` and `asSliceMut()` for maximum performance:

```zig
// ✅ GOOD: Zero-copy access
const data = try arr.asSlice(f64);
for (data) |val| {
    // Process val
}

// ❌ BAD: Copying data
const new_arr = try py.PyArray(@This()).fromSlice(f64, data);
```

### Type Safety

Specify the correct Zig type to avoid runtime errors:

```zig
// Create float64 array
const arr = try py.PyArray(@This()).zeros(f64, &[_]usize{100});

// Access with matching type
const data = try arr.asSlice(f64);  // ✅ Correct

const wrong = try arr.asSlice(i32);  // ❌ Type mismatch - will error
```

### Batch Operations

Process arrays in batches for better cache locality:

```zig
pub fn process_large_array(args: struct { arr: py.PyArray(@This()) }) !void {
    const data = try args.arr.asSliceMut(f64);

    const batch_size = 1024;
    var i: usize = 0;

    while (i < data.len) : (i += batch_size) {
        const end = @min(i + batch_size, data.len);
        const batch = data[i..end];

        // Process batch
        for (batch) |*val| {
            val.* = some_expensive_operation(val.*);
        }
    }
}
```

## Error Handling

NumPy operations can fail. Always handle errors:

```zig
pub fn safe_reshape(args: struct {
    arr: py.PyArray(@This()),
    rows: usize,
    cols: usize
}) !py.PyArray(@This()) {
    const size = try args.arr.size();

    // Validate reshape is possible
    if (rows * cols != size) {
        return py.ValueError(@This()).raiseFmt(
            "Cannot reshape array of size {d} into shape ({d}, {d})",
            .{ size, rows, cols }
        );
    }

    return try args.arr.reshape(&[_]usize{ rows, cols });
}
```

## Type Compatibility

Currently, pyZ3's NumPy integration works best with these dtypes:

- **Integers**: `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`
- **Floats**: `f32`, `f64`
- **Boolean**: `bool`

Complex numbers (`complex64`, `complex128`) are defined but may require additional handling.

### Python-Zig Type Mapping

| Python (NumPy) | Zig Type |
|----------------|----------|
| `np.bool_` | `bool` |
| `np.int8` | `i8` |
| `np.uint8` | `u8` |
| `np.int16` | `i16` |
| `np.uint16` | `u16` |
| `np.int32` | `i32` |
| `np.uint32` | `u32` |
| `np.int64` | `i64` |
| `np.uint64` | `u64` |
| `np.float32` | `f32` |
| `np.float64` | `f64` |

## Complete Example

Here's a complete example combining multiple features:

```zig
const py = @import("pyz3");

const root = @This();

pub const Statistics = py.class(struct {
    pub const __doc__ = "Statistical analysis of NumPy arrays";
    const Self = @This();

    data: py.PyArray(root),

    pub fn __init__(self: *Self, args: struct { data: py.PyArray(root) }) !void {
        self.* = .{ .data = args.data };
    }

    pub fn summary(self: *const Self) !struct {
        count: usize,
        min: f64,
        max: f64,
        mean: f64,
        sum: f64
    } {
        return .{
            .count = try self.data.size(),
            .min = try self.data.min(f64),
            .max = try self.data.max(f64),
            .mean = try self.data.mean(f64),
            .sum = try self.data.sum(f64),
        };
    }

    pub fn standardize(self: *const Self) !py.PyArray(root) {
        const mean_val = try self.data.mean(f64);
        const data = try self.data.asSlice(f64);

        // Calculate standard deviation
        var variance: f64 = 0.0;
        for (data) |val| {
            const diff = val - mean_val;
            variance += diff * diff;
        }
        variance /= @as(f64, @floatFromInt(data.len));
        const std_dev = @sqrt(variance);

        // Create standardized array
        const result = try py.PyArray(root).zeros(f64, &[_]usize{data.len});
        const result_data = try result.asSliceMut(f64);

        for (data, result_data) |val, *r| {
            r.* = (val - mean_val) / std_dev;
        }

        return result;
    }
});

pub fn create_sample_data(args: struct { size: usize }) !py.PyArray(root) {
    return try py.PyArray(root).zeros(f64, &[_]usize{args.size});
}

comptime {
    py.rootmodule(root);
}
```

Usage from Python:

```python
import numpy as np
import mymodule

# Create data
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Create statistics object
stats = mymodule.Statistics(data)

# Get summary
summary = stats.summary()
print(f"Mean: {summary['mean']}")
print(f"Std Dev: {summary['std']}")

# Standardize data
standardized = stats.standardize()
print(standardized)
```

## Best Practices

1. **Always specify the root type**: Use `py.PyArray(@This())` or `py.PyArray(root)`
2. **Match dtypes**: Ensure Zig types match NumPy dtypes
3. **Handle errors**: NumPy operations can fail
4. **Use zero-copy**: Prefer `asSlice()` over creating new arrays
5. **Validate dimensions**: Check array shapes before operations
6. **Release references**: Let Zig handle memory management with `defer`

## See Also

- [Buffers Guide](_6_buffers.md) - Low-level buffer protocol
- [Memory Guide](_5_memory.md) - Memory management in pyZ3
- [Examples](../../example/numpy_example.zig) - Complete NumPy examples
- [Tests](../../test/test_numpy.py) - NumPy integration tests
