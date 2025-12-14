// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! NumPy C API Bindings
//!
//! This module provides low-level Zig bindings to NumPy's C API for
//! high-performance array operations without Python overhead.
//!
//! Benefits over the Python API approach:
//! - Direct C-level array access
//! - No Python call overhead
//! - Better performance for array operations
//! - Access to more advanced NumPy C API features

const std = @import("std");
const ffi = @import("ffi");

/// NumPy type enumeration matching NPY_TYPES from ndarraytypes.h
pub const NPY_TYPES = enum(c_int) {
    NPY_BOOL = 0,
    NPY_BYTE = 1,
    NPY_UBYTE = 2,
    NPY_SHORT = 3,
    NPY_USHORT = 4,
    NPY_INT = 5,
    NPY_UINT = 6,
    NPY_LONG = 7,
    NPY_ULONG = 8,
    NPY_LONGLONG = 9,
    NPY_ULONGLONG = 10,
    NPY_FLOAT = 11,
    NPY_DOUBLE = 12,
    NPY_LONGDOUBLE = 13,
    NPY_CFLOAT = 14,
    NPY_CDOUBLE = 15,
    NPY_CLONGDOUBLE = 16,
    NPY_OBJECT = 17,
    NPY_STRING = 18,
    NPY_UNICODE = 19,
    NPY_VOID = 20,
    NPY_DATETIME = 21,
    NPY_TIMEDELTA = 22,
    NPY_HALF = 23,
    NPY_NTYPES_LEGACY = 24,
    NPY_NOTYPE = 25,
    NPY_USERDEF = 256,

    /// Get NPY_TYPES from Zig type at compile time
    pub fn fromZigType(comptime T: type) NPY_TYPES {
        return switch (T) {
            bool => .NPY_BOOL,
            i8 => .NPY_BYTE,
            u8 => .NPY_UBYTE,
            i16 => .NPY_SHORT,
            u16 => .NPY_USHORT,
            i32, c_int => .NPY_INT,
            u32, c_uint => .NPY_UINT,
            i64, c_long, isize => .NPY_LONG,
            u64, c_ulong, usize => .NPY_ULONG,
            f32 => .NPY_FLOAT,
            f64 => .NPY_DOUBLE,
            else => @compileError("Unsupported NumPy type for: " ++ @typeName(T)),
        };
    }
};

/// NumPy array flags matching ndarraytypes.h
pub const NPY_ARRAY_FLAGS = struct {
    pub const C_CONTIGUOUS: c_int = 0x0001;
    pub const F_CONTIGUOUS: c_int = 0x0002;
    pub const OWNDATA: c_int = 0x0004;
    pub const FORCECAST: c_int = 0x0010;
    pub const ENSURECOPY: c_int = 0x0020;
    pub const ENSUREARRAY: c_int = 0x0040;
    pub const ELEMENTSTRIDES: c_int = 0x0080;
    pub const ALIGNED: c_int = 0x0100;
    pub const NOTSWAPPED: c_int = 0x0200;
    pub const WRITEABLE: c_int = 0x0400;
    pub const UPDATEIFCOPY: c_int = 0x1000;
    pub const WRITEBACKIFCOPY: c_int = 0x2000;

    // Combinations
    pub const BEHAVED: c_int = ALIGNED | WRITEABLE;
    pub const BEHAVED_NS: c_int = ALIGNED | WRITEABLE | NOTSWAPPED;
    pub const CARRAY: c_int = C_CONTIGUOUS | BEHAVED;
    pub const CARRAY_RO: c_int = C_CONTIGUOUS | ALIGNED;
    pub const FARRAY: c_int = F_CONTIGUOUS | BEHAVED;
    pub const FARRAY_RO: c_int = F_CONTIGUOUS | ALIGNED;
    pub const DEFAULT: c_int = CARRAY;
    pub const IN_ARRAY: c_int = CARRAY_RO;
    pub const OUT_ARRAY: c_int = CARRAY;
    pub const INOUT_ARRAY: c_int = CARRAY | UPDATEIFCOPY;
};

/// Opaque PyArrayObject type
pub const PyArrayObject = opaque {};

/// NumPy C API function pointers
/// These will be initialized via import_array()
pub const CAPI = struct {
    var initialized: bool = false;

    /// Check if an object is a NumPy array
    pub extern fn PyArray_Check(obj: *ffi.PyObject) c_int;

    /// Get number of dimensions
    pub extern fn PyArray_NDIM(arr: *PyArrayObject) c_int;

    /// Get array dimensions (shape)
    pub extern fn PyArray_DIMS(arr: *PyArrayObject) [*c]isize;

    /// Get array strides
    pub extern fn PyArray_STRIDES(arr: *PyArrayObject) [*c]isize;

    /// Get array data pointer
    pub extern fn PyArray_DATA(arr: *PyArrayObject) ?*anyopaque;

    /// Get array item size in bytes
    pub extern fn PyArray_ITEMSIZE(arr: *PyArrayObject) c_int;

    /// Get array type number
    pub extern fn PyArray_TYPE(arr: *PyArrayObject) c_int;

    /// Get array flags
    pub extern fn PyArray_FLAGS(arr: *PyArrayObject) c_int;

    /// Create new array from descriptor
    pub extern fn PyArray_NewFromDescr(
        subtype: *ffi.PyTypeObject,
        descr: *anyopaque,
        nd: c_int,
        dims: [*c]isize,
        strides: [*c]isize,
        data: ?*anyopaque,
        flags: c_int,
        obj: ?*ffi.PyObject,
    ) ?*PyArrayObject;

    /// Create array from any object
    pub extern fn PyArray_FromAny(
        op: *ffi.PyObject,
        newtype: ?*anyopaque,
        min_depth: c_int,
        max_depth: c_int,
        requirements: c_int,
        context: ?*ffi.PyObject,
    ) ?*PyArrayObject;

    /// Get array descriptor from type
    pub extern fn PyArray_DescrFromType(type_num: c_int) ?*anyopaque;

    /// Create zeros array
    pub extern fn PyArray_Zeros(
        nd: c_int,
        dims: [*c]isize,
        dtype: *anyopaque,
        fortran: c_int,
    ) ?*PyArrayObject;

    /// Create ones array (helper function implemented via zeros + fill)
    pub fn PyArray_Ones(
        nd: c_int,
        dims: [*c]isize,
        dtype: *anyopaque,
        fortran: c_int,
    ) ?*PyArrayObject {
        // Create zeros array first
        const arr = PyArray_Zeros(nd, dims, dtype, fortran) orelse return null;

        // Get array info
        const arr_data = PyArray_DATA(arr) orelse return null;
        const itemsize = PyArray_ITEMSIZE(arr);
        const type_num = PyArray_TYPE(arr);

        // Calculate total number of elements
        var total_size: usize = 1;
        for (0..@intCast(nd)) |i| {
            total_size *= @intCast(dims[i]);
        }

        // Fill with ones based on dtype
        // This is type-specific because different dtypes have different representations of "1"
        const byte_ptr: [*]u8 = @ptrCast(arr_data);

        switch (type_num) {
            @intFromEnum(NPY_TYPES.NPY_FLOAT) => {
                const ptr: [*]f32 = @ptrCast(@alignCast(arr_data));
                for (0..total_size) |i| ptr[i] = 1.0;
            },
            @intFromEnum(NPY_TYPES.NPY_DOUBLE) => {
                const ptr: [*]f64 = @ptrCast(@alignCast(arr_data));
                for (0..total_size) |i| ptr[i] = 1.0;
            },
            @intFromEnum(NPY_TYPES.NPY_INT), @intFromEnum(NPY_TYPES.NPY_LONG) => {
                const ptr: [*]c_long = @ptrCast(@alignCast(arr_data));
                for (0..total_size) |i| ptr[i] = 1;
            },
            @intFromEnum(NPY_TYPES.NPY_UINT), @intFromEnum(NPY_TYPES.NPY_ULONG) => {
                const ptr: [*]c_ulong = @ptrCast(@alignCast(arr_data));
                for (0..total_size) |i| ptr[i] = 1;
            },
            @intFromEnum(NPY_TYPES.NPY_BOOL) => {
                const ptr: [*]u8 = @ptrCast(arr_data);
                for (0..total_size) |i| ptr[i] = 1;
            },
            else => {
                // For unsupported types, fill bytes with 0x01
                // This is a fallback and may not be correct for all types
                @memset(byte_ptr[0..(total_size * @as(usize, @intCast(itemsize)))], 1);
            },
        }

        return arr;
    }

    /// Copy array
    pub extern fn PyArray_Copy(arr: *PyArrayObject) ?*PyArrayObject;

    /// Reshape array
    pub extern fn PyArray_Reshape(arr: *PyArrayObject, shape: *ffi.PyObject) ?*PyArrayObject;

    /// Transpose array
    pub extern fn PyArray_Transpose(arr: *PyArrayObject, permute: ?*ffi.PyObject) ?*PyArrayObject;

    /// Initialize NumPy C API
    /// Must be called before using any C API functions
    ///
    /// NOTE: Full import_array() initialization requires NumPy headers at compile time.
    /// This function performs basic initialization by ensuring NumPy is loaded via Python.
    /// For production use with full C API access, compile with NumPy headers included.
    pub fn initialize() !void {
        if (initialized) return;

        // Ensure NumPy is loaded by importing it via Python API
        // This is a hybrid approach that works without compile-time NumPy headers
        const numpy_module = ffi.PyImport_ImportModule("numpy") orelse {
            return error.NumPyNotAvailable;
        };
        defer ffi.Py_DecRef(numpy_module);

        // Get NumPy version to verify it loaded correctly
        const version_obj = ffi.PyObject_GetAttrString(numpy_module, "__version__") orelse {
            return error.NumPyVersionCheckFailed;
        };
        defer ffi.Py_DecRef(version_obj);

        // At this point, NumPy's C API symbols should be available in the process
        // import_array() would be called here if we had NumPy headers at compile time
        // For now, we rely on the fact that importing numpy makes symbols available

        // TODO: When NumPy headers are available at build time, uncomment:
        // if (import_array() < 0) {
        //     return error.ImportArrayFailed;
        // }

        initialized = true;
    }

    pub fn isInitialized() bool {
        return initialized;
    }
};

/// Helper functions for working with NumPy C API

/// Check if a Python object is a NumPy array
pub fn isArray(obj: *ffi.PyObject) bool {
    if (!CAPI.initialized) {
        CAPI.initialize() catch return false;
    }
    return CAPI.PyArray_Check(obj) != 0;
}

/// Get array shape as Zig slice
pub fn getShape(arr: *PyArrayObject, allocator: std.mem.Allocator) ![]usize {
    const ndim = CAPI.PyArray_NDIM(arr);
    const dims = CAPI.PyArray_DIMS(arr);

    const shape = try allocator.alloc(usize, @intCast(ndim));
    for (0..@intCast(ndim)) |i| {
        shape[i] = @intCast(dims[i]);
    }

    return shape;
}

/// Get array data as Zig slice (zero-copy)
pub fn getData(comptime T: type, arr: *PyArrayObject) ![]T {
    // Verify type matches
    const expected_type = NPY_TYPES.fromZigType(T);
    const actual_type = CAPI.PyArray_TYPE(arr);

    if (@intFromEnum(expected_type) != actual_type) {
        return error.TypeMismatch;
    }

    // Check if array is C-contiguous
    const flags = CAPI.PyArray_FLAGS(arr);
    if ((flags & NPY_ARRAY_FLAGS.C_CONTIGUOUS) == 0) {
        return error.NotContiguous;
    }

    // Get data pointer and size
    const data_ptr = CAPI.PyArray_DATA(arr) orelse return error.NullData;
    const ndim = CAPI.PyArray_NDIM(arr);
    const dims = CAPI.PyArray_DIMS(arr);

    // Calculate total size
    var total_size: usize = 1;
    for (0..@intCast(ndim)) |i| {
        total_size *= @intCast(dims[i]);
    }

    const ptr: [*]T = @ptrCast(@alignCast(data_ptr));
    return ptr[0..total_size];
}

/// Get mutable array data as Zig slice (zero-copy)
pub fn getDataMut(comptime T: type, arr: *PyArrayObject) ![]T {
    // Check if writeable
    const flags = CAPI.PyArray_FLAGS(arr);
    if ((flags & NPY_ARRAY_FLAGS.WRITEABLE) == 0) {
        return error.NotWriteable;
    }

    return getData(T, arr);
}

/// Get NumPy ndarray type object (cached)
var cached_ndarray_type: ?*ffi.PyTypeObject = null;

fn getNdArrayType() !*ffi.PyTypeObject {
    if (cached_ndarray_type) |t| return t;

    // Get via Python API
    const numpy_module = ffi.PyImport_ImportModule("numpy") orelse {
        return error.NumPyNotAvailable;
    };
    defer ffi.Py_DecRef(numpy_module);

    const ndarray_obj = ffi.PyObject_GetAttrString(numpy_module, "ndarray") orelse {
        return error.NdArrayTypeNotFound;
    };

    // Cast to PyTypeObject and cache it
    // We don't decref because we're caching it
    const type_obj: *ffi.PyTypeObject = @ptrCast(@alignCast(ndarray_obj));
    cached_ndarray_type = type_obj;

    return type_obj;
}

/// Create a new NumPy array from Zig slice
pub fn fromSlice(comptime T: type, data: []const T, allocator: std.mem.Allocator) !*PyArrayObject {
    _ = allocator;

    if (!CAPI.initialized) {
        try CAPI.initialize();
    }

    const type_num = NPY_TYPES.fromZigType(T);
    const descr = CAPI.PyArray_DescrFromType(@intFromEnum(type_num)) orelse return error.DescrCreationFailed;

    var dims = [_]isize{@intCast(data.len)};

    // Get ndarray type via Python API (hybrid approach)
    const ndarray_type = try getNdArrayType();

    // Create new array
    const arr = CAPI.PyArray_NewFromDescr(
        ndarray_type,
        descr,
        1, // 1D array
        &dims,
        null, // strides
        null, // data (allocate new)
        NPY_ARRAY_FLAGS.DEFAULT,
        null, // obj
    ) orelse return error.ArrayCreationFailed;

    // Copy data
    const arr_data = try getDataMut(T, arr);
    @memcpy(arr_data, data);

    return arr;
}

/// Create zeros array
pub fn zeros(comptime T: type, shape: []const usize, allocator: std.mem.Allocator) !*PyArrayObject {
    if (!CAPI.initialized) {
        try CAPI.initialize();
    }

    const type_num = NPY_TYPES.fromZigType(T);
    const descr = CAPI.PyArray_DescrFromType(@intFromEnum(type_num)) orelse return error.DescrCreationFailed;

    // Convert shape to isize array
    var dims = try allocator.alloc(isize, shape.len);
    defer allocator.free(dims);

    for (shape, 0..) |s, i| {
        dims[i] = @intCast(s);
    }

    const arr = CAPI.PyArray_Zeros(
        @intCast(shape.len),
        dims.ptr,
        descr,
        0, // C-order
    ) orelse return error.ArrayCreationFailed;

    return arr;
}

/// Documentation and usage examples
pub const docs = struct {
    pub const overview =
        \\NumPy C API Integration
        \\
        \\This module provides low-level C API bindings for NumPy, enabling:
        \\- Direct C-level array manipulation
        \\- Zero-copy data access
        \\- Better performance than Python API
        \\- Access to advanced NumPy features
        \\
        \\Example usage:
        \\
        \\  const np = @import("numpy_capi.zig");
        \\
        \\  // Initialize C API
        \\  try np.CAPI.initialize();
        \\
        \\  // Check if object is array
        \\  if (np.isArray(obj)) {
        \\      const arr = @ptrCast(*np.PyArrayObject, obj);
        \\
        \\      // Get data as Zig slice
        \\      const data = try np.getData(f64, arr);
        \\
        \\      // Process data
        \\      for (data) |val| {
        \\          // ...
        \\      }
        \\  }
    ;
};

// NOTE: PyArray_Type is obtained dynamically via getNdArrayType() instead of extern declaration
// This avoids needing to link against NumPy at compile time
