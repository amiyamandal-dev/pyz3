# Complete Guide: Opaque Zig State in Python Classes

## Overview

This guide explains how pyz3 enables Python classes to hold completely opaque Zig-native state (allocators, pointers, structs, handles) that is:

- ✅ Attached at `__init__`
- ✅ Accessible only within Zig methods
- ✅ Completely invisible to Python code
- ✅ Safely cleaned up in `__del__`

## How It Works: The PyTypeStruct Pattern

### Memory Layout

When you define a pyz3 class:

```zig
pub const MyClass = py.class(struct {
    allocator: std.mem.Allocator,
    buffer: []u8,
    count: usize,
});
```

pyz3 creates a `PyTypeStruct` wrapper:

```
┌──────────────────────────────┐
│ ffi.PyObject                │ ← CPython header (Python sees this pointer)
│ - ob_refcnt                 │
│ - ob_type                   │
├──────────────────────────────┤
│ YourZigState                │ ← Your struct (OPAQUE to Python)
│ - allocator                 │
│ - buffer                    │
│ - count                     │
└──────────────────────────────┘
```

**Key insight:** Python receives a pointer to `ffi.PyObject`, but has NO mechanism to access the Zig state that follows it in memory.

### Type Registration

File: `pyz3/src/pytypes.zig:66-74`

```zig
const spec = ffi.PyType_Spec{
    .name = qualifiedName.ptr,
    .basicsize = @sizeOf(PyTypeStruct(definition)),  // Includes state!
    .itemsize = 0,
    .flags = flags,
    .slots = @constCast(slots.slots.ptr),
};
```

CPython allocates `basicsize` bytes, but only knows about the `PyObject` header. The rest is your opaque Zig state.

## Requirement 1: Opaque State Attachment

### Pattern

```zig
pub const SecureVault = py.class(struct {
    const Self = @This();

    // ALL fields are opaque to Python
    allocator: std.mem.Allocator,
    encryption_key: [32]u8,
    encrypted_data: []u8,

    pub fn __init__(self: *Self) !void {
        self.allocator = py.allocator;
        self.encrypted_data = try self.allocator.alloc(u8, 0);
        std.crypto.random.bytes(&self.encryption_key);
    }
});
```

### Why Python Cannot Access It

1. **No `__dict__` entries:** CPython classes implemented in C don't use `__dict__` for data storage
2. **No PyMemberDef:** State fields are not registered as Python members
3. **No getters/setters:** No `tp_getattro` that would expose fields
4. **Memory layout:** Python can't even find the offset (it only knows `PyObject` size)

**Proof:**
```python
>>> vault = SecureVault()
>>> vault.encryption_key  # AttributeError
>>> vault.__dict__  # {} or AttributeError
>>> vars(vault)  # {} or TypeError
>>> dir(vault)  # Only shows methods, not fields
```

## Requirement 2: Zig-Only Accessibility

### How Zig Methods Access State

File: `pyz3/src/pytypes.zig:298-304`

When a Python method is called, the CPython slot function receives `PyObject*`:

```zig
fn method_wrapper(pyself: *ffi.PyObject, pyargs: ...) callconv(.c) ... {
    // Cast PyObject* back to full struct
    const self = tramp.Trampoline(root, sig.selfParam.?).unwrap(
        py.PyObject{ .py = pyself }
    ) catch return error;

    // Now self is *YourStruct with full access
    definition.your_method(self, args);
}
```

### Type Casting Implementation

File: `pyz3/src/conversions.zig:92-101`

```zig
pub inline fn unchecked(comptime root: type, comptime T: type, obj: py.PyObject) T {
    const Definition = @typeInfo(T).pointer.child;
    // Cast to full PyTypeStruct to access state
    const instance: *pytypes.PyTypeStruct(Definition) = @ptrCast(@alignCast(obj.py));
    return &instance.state;
}
```

This is safe because:
- Only Zig code can perform the cast
- Python has no casting mechanism
- The memory layout is guaranteed by `basicsize`

## Requirement 3: Non-Native Data Structures

### Supported Opaque Types

All Zig types are supported:

```zig
pub const ComplexClass = py.class(struct {
    // Allocators
    arena: std.heap.ArenaAllocator,
    gpa: std.heap.GeneralPurposeAllocator(.{}),

    // Pointers & slices
    buffer: []u8,
    context: *MyContext,

    // Structs
    config: Config,

    // Collections
    map: std.AutoHashMap(u64, *Node),
    pool: std.mem.Pool(Node),

    // OS handles
    file: ?std.fs.File,
    fd: ?std.posix.fd_t,

    // Synchronization
    mutex: std.Thread.Mutex,
    condition: std.Thread.Condition,
});
```

Python cannot represent any of these, so they remain Zig-only.

## Requirement 4: Deterministic Cleanup

### __del__ Implementation

File: `pyz3/src/pytypes.zig:327-340`

```zig
fn tp_finalize(pyself: *ffi.PyObject) callconv(.c) void {
    // Save exception state (CPython requirement)
    var error_type: ?*ffi.PyObject = undefined;
    var error_value: ?*ffi.PyObject = undefined;
    var error_tb: ?*ffi.PyObject = undefined;
    ffi.PyErr_Fetch(&error_type, &error_value, &error_tb);

    // Call user's __del__
    const instance: *PyTypeStruct(definition) = @ptrCast(pyself);
    definition.__del__(&instance.state);

    // Restore exception state
    ffi.PyErr_Restore(error_type, error_value, error_tb);
}
```

**Guarantees:**
- ✅ Called exactly once by CPython garbage collector
- ✅ Exception state preserved (no crashes)
- ✅ All Zig state accessible for cleanup

### Double-Free Prevention Pattern

```zig
pub const SafeClass = py.class(struct {
    allocator: std.mem.Allocator,
    buffer: ?[]u8,
    cleaned_up: bool,

    pub fn __del__(self: *Self) void {
        if (self.cleaned_up) return;

        if (self.buffer) |buf| {
            self.allocator.free(buf);
            self.buffer = null;
        }

        self.cleaned_up = true;
    }
});
```

### Manual Cleanup Pattern

```zig
pub fn cleanup(self: *Self) void {
    if (self.cleaned_up) return;

    // Free all resources
    self.allocator.free(self.buffer);
    if (self.file) |f| f.close();

    self.cleaned_up = true;
}

pub fn __del__(self: *Self) void {
    self.cleanup();  // Safe to call multiple times
}
```

## Requirement 5: Python Safety Guarantees

### What Python CANNOT Do

1. **Access internal state:**
   ```python
   >>> obj.allocator  # AttributeError
   >>> obj.buffer     # AttributeError
   ```

2. **Leak allocator or pointer:**
   - No way to obtain a reference to internal state
   - No capsule objects created
   - No memory addresses exposed

3. **Influence lifetime:**
   - `__del__` timing controlled by CPython GC
   - Manual `__del__()` call is safe but doesn't affect actual cleanup
   - Reference counting handled by CPython

4. **See internal state via introspection:**
   ```python
   >>> import inspect
   >>> inspect.getmembers(obj)  # Only methods, no data fields
   >>> dir(obj)  # Only methods
   >>> vars(obj)  # {}
   ```

### Automatic Garbage Collection Integration

File: `pyz3/src/pytypes.zig:477-651`

pyz3 automatically detects if your class contains `PyObject` references:

```zig
const needsGc = classNeedsGc(definition);

fn classNeedsGc(comptime CT: type) bool {
    inline for (@typeInfo(CT).@"struct".fields) |field| {
        if (typeNeedsGc(field.type)) {
            return true;
        }
    }
    return false;
}
```

If true, `tp_traverse` and `tp_clear` are automatically registered for cycle detection.

### Memory Safety Checklist

- ✅ **Use `py.allocator`** for all dynamic memory
- ✅ **Implement `__del__`** for cleanup
- ✅ **Add cleanup guards** to prevent double-free
- ✅ **Store, don't borrow** PyObject references (use `incref()`)
- ✅ **Return copies** of internal data, never pointers
- ✅ **Test with ASAN** to detect leaks

## Complete Example

See `example/opaque_state.zig` for 5 complete examples:

1. **BufferManager** - Simple allocator + buffer
2. **DataProcessor** - Complex arena allocator
3. **SecureStorage** - Encryption key (security-critical)
4. **FileManager** - File handle management
5. **SharedResource** - Reference counting

## Testing Opaqueness

See `test/test_opaque_state.py` for comprehensive tests proving:

- ✅ No attribute access to internal state
- ✅ No `__dict__` exposure
- ✅ No `vars()` exposure
- ✅ No `dir()` exposure
- ✅ No `inspect` exposure
- ✅ Deterministic cleanup
- ✅ No memory leaks
- ✅ Double-free prevention

## Architecture Comparison

### Python Class (Pure Python)

```python
class PyClass:
    def __init__(self):
        self.data = []  # ❌ Visible in __dict__
        self._private = 0  # ❌ Still accessible
```

```python
>>> obj = PyClass()
>>> obj.__dict__  # {'data': [], '_private': 0}
>>> obj._private  # 0 (accessible!)
```

### pyz3 Class (Zig Implementation)

```zig
pub const ZigClass = py.class(struct {
    data: []u8,
    private: usize,
});
```

```python
>>> obj = ZigClass()
>>> obj.__dict__  # {}
>>> obj.data  # AttributeError ✅
>>> obj.private  # AttributeError ✅
```

## Best Practices

### ✅ DO

1. **Store all internal state in the class struct**
   ```zig
   pub const Good = py.class(struct {
       allocator: std.mem.Allocator,  // ✅ Opaque
       buffer: []u8,                   // ✅ Opaque
   });
   ```

2. **Use `py.allocator` for persistent allocations**
   ```zig
   pub fn __init__(self: *Self) !void {
       self.allocator = py.allocator;  // ✅ Use framework allocator
       self.buffer = try self.allocator.alloc(u8, 1024);
   }
   ```

3. **Implement `__del__` for cleanup**
   ```zig
   pub fn __del__(self: *Self) void {
       self.allocator.free(self.buffer);  // ✅ Always cleanup
   }
   ```

4. **Add double-free guards**
   ```zig
   cleaned_up: bool = false,

   pub fn __del__(self: *Self) void {
       if (self.cleaned_up) return;  // ✅ Safe
       // ... cleanup
       self.cleaned_up = true;
   }
   ```

### ❌ DON'T

1. **Don't expose internal pointers**
   ```zig
   pub fn get_buffer(self: *Self) []u8 {
       return self.buffer;  // ❌ Returns internal reference
   }

   // ✅ Better: return copy
   pub fn get_buffer_copy(self: *Self) !py.PyBytes {
       return py.PyBytes.create(self.buffer);
   }
   ```

2. **Don't leak allocators**
   ```zig
   pub fn __init__(self: *Self) !void {
       const allocator = std.heap.page_allocator;  // ❌ Lost reference
       self.buffer = try allocator.alloc(...);
   }

   // ✅ Better: store allocator
   pub fn __init__(self: *Self) !void {
       self.allocator = py.allocator;
       self.buffer = try self.allocator.alloc(...);
   }
   ```

3. **Don't assume `__del__` timing**
   ```zig
   // ❌ Bad: relies on immediate cleanup
   def process_file(path):
       obj = FileManager()
       obj.open(path)
       obj.write(data)
       # File might not be closed yet!

   // ✅ Better: explicit cleanup
   def process_file(path):
       obj = FileManager()
       try:
           obj.open(path)
           obj.write(data)
       finally:
           obj.close()  # Deterministic
   ```

## Summary

pyz3's `PyTypeStruct` pattern provides:

✅ **Complete opacity** - Python cannot access Zig state
✅ **Type safety** - Only Zig code can cast to internal types
✅ **Memory safety** - GIL-aware allocator, GC integration
✅ **Deterministic cleanup** - `__del__` called exactly once
✅ **Zero overhead** - Direct struct access in Zig

This enables building high-performance, secure Python extensions with complete control over internal state while maintaining Python's safety guarantees.

## References

- **Core implementation:** `pyz3/src/pytypes.zig`
- **Type conversion:** `pyz3/src/conversions.zig`
- **Memory allocator:** `pyz3/src/mem.zig`
- **Examples:** `example/opaque_state.zig`
- **Tests:** `test/test_opaque_state.py`
- **Class examples:** `example/classes.zig`, `example/buffers.zig`
