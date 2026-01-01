# Memory-Mapped Files (mmap)

pyz3 provides utilities for memory-mapped file I/O, enabling zero-copy data access and efficient inter-process communication.

## Overview

Memory-mapped files allow you to:
- **Zero-copy I/O**: Access large files without loading them entirely into RAM
- **Shared memory**: Share data between processes without serialization
- **Lazy loading**: Pages are loaded on-demand by the OS
- **Fast random access**: Efficient access to any part of the file

## Python API

### MmapArray

The primary class for working with memory-mapped NumPy arrays:

```python
from pyz3.mmap_utils import MmapArray
import numpy as np

# Create a new mmap array
arr = MmapArray.create("data.bin", shape=(1000000,), dtype=np.float64)
arr[:] = np.random.randn(1000000)
arr.flush()  # Ensure data is written to disk

# Later, open existing array (zero-copy!)
arr = MmapArray("data.bin", shape=(1000000,), dtype=np.float64, mode="r")
print(f"Mean: {arr.mean()}")  # Data loaded on-demand
arr.close()
```

#### Creating from Existing Array

```python
# Save an existing numpy array to mmap
original = np.random.randn(1000000)
mmap_arr = MmapArray.from_array("weights.bin", original)

# Now accessible as mmap
print(mmap_arr.sum())
```

#### Context Manager

```python
with MmapArray.create("temp.bin", shape=(100,), dtype=np.float32) as arr:
    arr[:] = np.arange(100)
    # File automatically closed when exiting context
```

#### Array-Like Interface

MmapArray supports common NumPy operations:

```python
with MmapArray("data.bin", shape=(1000,), dtype=np.float64) as arr:
    # Slicing
    subset = arr[100:200]

    # Aggregations
    print(arr.sum(), arr.mean(), arr.std())
    print(arr.min(), arr.max())

    # Length
    print(len(arr))

    # Use as numpy array
    result = np.sin(arr.array)
```

### SharedMemoryArray

For inter-process communication:

```python
from pyz3.mmap_utils import SharedMemoryArray
import multiprocessing

# Parent process: create shared memory
shared = SharedMemoryArray.create(shape=(1000,), dtype=np.float64)
shared[:] = np.zeros(1000)
shared.flush()

def worker(path, shape, dtype):
    """Worker process modifies shared data."""
    arr = SharedMemoryArray.open(path, shape, dtype)
    arr[:] = np.arange(1000)
    arr.flush()
    arr.close()

# Start worker
p = multiprocessing.Process(target=worker, args=(shared.path, (1000,), np.float64))
p.start()
p.join()

# Parent sees worker's changes (zero-copy!)
print(shared[500])  # 500.0
shared.unlink()  # Clean up
```

### Binary File Utilities

Quick functions for binary data:

```python
from pyz3.mmap_utils import save_binary_mmap, load_binary_mmap

# Save array to binary file
weights = np.random.randn(10000000)
save_binary_mmap("model_weights.bin", weights)

# Load with zero-copy mmap
loaded = load_binary_mmap("model_weights.bin", dtype=np.float64)
print(loaded.mean())

# Load with explicit shape
loaded_2d = load_binary_mmap("matrix.bin", dtype=np.float32, shape=(1000, 1000))

# Skip header bytes
data = load_binary_mmap("file_with_header.bin", dtype=np.float64, offset=128)
```

### MmapCache

LRU cache for frequently accessed mmap files:

```python
from pyz3.mmap_utils import MmapCache

cache = MmapCache(max_size=10)

# First access opens file
arr1 = cache.get_or_create("data.bin", shape=(1000,), dtype=np.float64)

# Second access returns cached object
arr2 = cache.get_or_create("data.bin", shape=(1000,), dtype=np.float64)
assert arr1 is arr2  # Same object!

# Cache automatically evicts LRU entries when full
cache.clear()  # Manual cleanup
```

### Raw mmap Access

For non-array data:

```python
from pyz3.mmap_utils import mmap_file

with mmap_file("data.bin", mode="r") as mm:
    # Read bytes directly
    header = mm[:4]
    chunk = mm[100:200]

with mmap_file("data.bin", mode="r+") as mm:
    # Modify in place
    mm[0:5] = b"HELLO"
```

## Zig API

### MmapFile

```zig
const py = @import("pyz3");
const mmap = py.mmap;

// Open existing file
var file = try mmap.MmapFile.open("data.bin", .{ .read = true, .write = true });
defer file.close();

// Access as typed slice
const floats = try file.asSliceMut(f64);
for (floats) |*f| {
    f.* *= 2.0;
}

try file.flush();
```

### Creating Files

```zig
// Create new mmap file
var file = try mmap.MmapFile.create("output.bin", 1024 * 1024);  // 1MB
defer file.close();

var data = try file.asSliceMut(u8);
@memset(data, 0);

try file.flush();
```

### Access Hints

```zig
var file = try mmap.MmapFile.open("data.bin", .{ .read = true });
defer file.close();

// Hint to kernel about access pattern
file.advise(.sequential);  // Reading sequentially
// or
file.advise(.random);      // Random access
file.advise(.willneed);    // Prefetch data
file.advise(.dontneed);    // Done with data
```

### Shared Buffers

```zig
// Create anonymous shared memory
const result = try mmap.createSharedBuffer(f64, 1000);
defer result.mmap.close();

// Use the slice
for (result.slice, 0..) |*val, i| {
    val.* = @floatFromInt(i);
}
```

## Use Cases

### 1. Large Dataset Processing

```python
# Load 10GB dataset without using 10GB RAM
dataset = MmapArray("huge_dataset.bin", shape=(1_000_000_000,), dtype=np.float32)

# Process in chunks - only loads needed pages
for i in range(0, len(dataset), 1_000_000):
    chunk = dataset[i:i+1_000_000]
    process(chunk)
```

### 2. Model Weight Loading

```python
# Traditional: loads entire file into RAM
weights = np.load("model.npy")  # 8GB in RAM

# With mmap: lazy loading, shared across processes
weights = load_binary_mmap("model.bin", dtype=np.float32, shape=(1000000000,))
# Only accessed pages loaded into RAM
```

### 3. Multi-Process Data Sharing

```python
# Parent creates shared training data
data = SharedMemoryArray.create(shape=(1000000, 784), dtype=np.float32)
data[:] = load_training_data()
data.flush()

# Spawn workers - they share the same memory
for i in range(num_workers):
    p = Process(target=train_worker, args=(data.path, data.shape, data.dtype))
    p.start()

# No data copying between processes!
```

### 4. Real-time Data Pipeline

```python
# Producer writes to mmap
producer = SharedMemoryArray.create(shape=(buffer_size,), dtype=np.float64)

def producer_loop():
    while True:
        data = get_sensor_data()
        producer[:len(data)] = data
        producer.flush()

# Consumer reads from same mmap
consumer = SharedMemoryArray.open(producer.path, producer.shape, producer.dtype)

def consumer_loop():
    while True:
        process(consumer.array)
```

## Performance Tips

1. **Use sequential access hints** when reading files linearly
2. **Flush periodically** for durability, not after every write
3. **Choose appropriate page sizes** - OS typically uses 4KB pages
4. **Avoid sparse random writes** - can cause excessive page faults
5. **Unlink shared memory** when done to free resources

## Platform Notes

| Platform | Notes |
|----------|-------|
| Linux | Full support, best performance with `madvise` hints |
| macOS | Full support |
| Windows | Supported via different API (CreateFileMapping) |

## Comparison with Alternatives

| Method | Copy? | Shared? | Lazy? | Best For |
|--------|-------|---------|-------|----------|
| `np.load()` | Yes | No | No | Small files |
| `np.memmap()` | No | No | Yes | Single process |
| `MmapArray` | No | Yes | Yes | Large arrays |
| `SharedMemoryArray` | No | Yes | Yes | Multi-process |
| `pickle` | Yes | No | No | Complex objects |

## Troubleshooting

### "Bus error" on write
- Ensure file was opened with write mode
- Verify file has sufficient size

### Slow first access
- First access triggers page fault - normal behavior
- Use `madvise(WILLNEED)` to prefetch

### Memory usage seems high
- mmap counts toward virtual memory, not RSS
- Kernel may keep pages cached even after close
- This is normal OS behavior

### Changes not visible in other process
- Call `flush()` after writing
- Reopen file in other process if needed
