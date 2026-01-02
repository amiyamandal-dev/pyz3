// NumPy C API Zig bindings for pyz3
// Provides direct access to NumPy arrays via the C API
//
// Licensed under the Apache License, Version 2.0

const std = @import("std");
const py = @import("pyz3.zig");
const ffi = @import("ffi");
const numpy_ffi = @import("numpy_ffi");
const PyError = @import("errors.zig").PyError;

// Re-export NumPy FFI types
pub const PyArrayObject = numpy_ffi.PyArrayObject;
pub const PyArrayObject_fields = numpy_ffi.PyArrayObject_fields;
pub const PyArray_Descr = numpy_ffi.PyArray_Descr;
pub const npy_intp = numpy_ffi.npy_intp;

// Re-export NumPy type enums
pub const NPY_TYPES = numpy_ffi.NPY_TYPES;
pub const NPY_ARRAY_FLAGS = numpy_ffi.NPY_ARRAY_FLAGS;

/// NumPy data type enumeration with Zig-friendly names
pub const DType = enum(c_int) {
    bool_ = numpy_ffi.NPY_BOOL,
    int8 = numpy_ffi.NPY_BYTE,
    uint8 = numpy_ffi.NPY_UBYTE,
    int16 = numpy_ffi.NPY_SHORT,
    uint16 = numpy_ffi.NPY_USHORT,
    int32 = numpy_ffi.NPY_INT,
    uint32 = numpy_ffi.NPY_UINT,
    int64 = numpy_ffi.NPY_LONGLONG,
    uint64 = numpy_ffi.NPY_ULONGLONG,
    float32 = numpy_ffi.NPY_FLOAT,
    float64 = numpy_ffi.NPY_DOUBLE,
    complex64 = numpy_ffi.NPY_CFLOAT,
    complex128 = numpy_ffi.NPY_CDOUBLE,

    /// Get dtype from Zig type at compile time
    pub fn fromType(comptime T: type) DType {
        return switch (T) {
            bool => .bool_,
            i8 => .int8,
            u8 => .uint8,
            i16 => .int16,
            u16 => .uint16,
            i32, c_int => .int32,
            u32, c_uint => .uint32,
            i64, c_long, c_longlong, isize => .int64,
            u64, c_ulong, c_ulonglong, usize => .uint64,
            f32 => .float32,
            f64 => .float64,
            else => @compileError("Unsupported NumPy dtype for type: " ++ @typeName(T)),
        };
    }

    /// Get the size in bytes of this dtype
    pub fn size(self: DType) usize {
        return switch (self) {
            .bool_, .int8, .uint8 => 1,
            .int16, .uint16 => 2,
            .int32, .uint32, .float32 => 4,
            .int64, .uint64, .float64, .complex64 => 8,
            .complex128 => 16,
        };
    }

    /// Get the dtype name string
    pub fn name(self: DType) []const u8 {
        return switch (self) {
            .bool_ => "bool",
            .int8 => "int8",
            .uint8 => "uint8",
            .int16 => "int16",
            .uint16 => "uint16",
            .int32 => "int32",
            .uint32 => "uint32",
            .int64 => "int64",
            .uint64 => "uint64",
            .float32 => "float32",
            .float64 => "float64",
            .complex64 => "complex64",
            .complex128 => "complex128",
        };
    }
};

/// NumPy Array wrapper providing direct C API access
/// This provides zero-copy access to NumPy array data
pub fn PyArray(comptime root: type) type {
    return struct {
        obj: py.PyObject,

        const Self = @This();

        /// Check if a Python object is a NumPy array
        pub fn check(obj: py.PyObject) bool {
            // Get numpy.ndarray type and check isinstance
            const np = py.import(root, "numpy") catch return false;
            defer np.decref();

            const ndarray_type = np.get("ndarray") catch return false;
            defer ndarray_type.decref();

            // Use PyObject_IsInstance from FFI
            const result = ffi.PyObject_IsInstance(obj.py, ndarray_type.py);
            return result == 1;
        }

        /// Create PyArray from a PyObject (checked)
        pub const from = struct {
            pub fn checked(obj: py.PyObject) PyError!Self {
                if (!check(obj)) {
                    return py.TypeError(root).raise("Expected numpy.ndarray");
                }
                return Self{ .obj = obj };
            }

            pub fn unchecked(obj: py.PyObject) Self {
                return Self{ .obj = obj };
            }
        };

        /// Get the underlying PyArrayObject pointer
        pub fn ptr(self: Self) *PyArrayObject {
            return @ptrCast(self.obj.py);
        }

        /// Increment reference count
        pub fn incref(self: Self) void {
            self.obj.incref();
        }

        /// Decrement reference count
        pub fn decref(self: Self) void {
            self.obj.decref();
        }

        // ====================================================================
        // Direct C API accessor functions (zero-copy)
        // ====================================================================

        /// Get number of dimensions
        pub fn ndim(self: Self) usize {
            return @intCast(numpy_ffi.pyz3_PyArray_NDIM(self.ptr()));
        }

        /// Get raw data pointer (zero-copy access)
        pub fn data(self: Self) ?*anyopaque {
            return numpy_ffi.pyz3_PyArray_DATA(self.ptr());
        }

        /// Get data as typed pointer (zero-copy)
        /// Returns null if data pointer is null or alignment doesn't match
        pub fn dataAs(self: Self, comptime T: type) ?[*]T {
            const ptr_data = self.data() orelse return null;
            // Check alignment - if not properly aligned, return null
            const addr = @intFromPtr(ptr_data);
            if (addr % @alignOf(T) != 0) {
                return null;
            }
            return @ptrCast(@alignCast(ptr_data));
        }

        /// Get shape array pointer
        pub fn dims(self: Self) [*]npy_intp {
            return numpy_ffi.pyz3_PyArray_DIMS(self.ptr());
        }

        /// Get shape as Zig slice (borrows from array)
        pub fn shape(self: Self) []const npy_intp {
            const nd = self.ndim();
            return self.dims()[0..nd];
        }

        /// Get strides array pointer
        pub fn stridesPtr(self: Self) [*]npy_intp {
            return numpy_ffi.pyz3_PyArray_STRIDES(self.ptr());
        }

        /// Get strides as Zig slice (borrows from array)
        pub fn strides(self: Self) []const npy_intp {
            const nd = self.ndim();
            return self.stridesPtr()[0..nd];
        }

        /// Get dimension at index
        pub fn dim(self: Self, idx: usize) npy_intp {
            return numpy_ffi.pyz3_PyArray_DIM(self.ptr(), @intCast(idx));
        }

        /// Get stride at index
        pub fn stride(self: Self, idx: usize) npy_intp {
            return numpy_ffi.pyz3_PyArray_STRIDE(self.ptr(), @intCast(idx));
        }

        /// Get item size in bytes
        pub fn itemsize(self: Self) usize {
            return @intCast(numpy_ffi.pyz3_PyArray_ITEMSIZE(self.ptr()));
        }

        /// Get array flags
        pub fn flags(self: Self) c_int {
            return numpy_ffi.pyz3_PyArray_FLAGS(self.ptr());
        }

        /// Get type number (NPY_TYPES enum value)
        pub fn typeNum(self: Self) c_int {
            return numpy_ffi.pyz3_PyArray_TYPE(self.ptr());
        }

        /// Get dtype as DType enum
        pub fn dtype(self: Self) ?DType {
            const type_num = self.typeNum();
            return std.meta.intToEnum(DType, type_num) catch null;
        }

        /// Get total number of elements
        pub fn size(self: Self) usize {
            return @intCast(numpy_ffi.pyz3_PyArray_SIZE(self.ptr()));
        }

        /// Get total number of bytes
        pub fn nbytes(self: Self) usize {
            return @intCast(numpy_ffi.pyz3_PyArray_NBYTES(self.ptr()));
        }

        // ====================================================================
        // Flag checks
        // ====================================================================

        /// Check if array is C-contiguous
        pub fn isCContiguous(self: Self) bool {
            return numpy_ffi.pyz3_PyArray_IS_C_CONTIGUOUS(self.ptr()) != 0;
        }

        /// Check if array is Fortran-contiguous
        pub fn isFContiguous(self: Self) bool {
            return numpy_ffi.pyz3_PyArray_IS_F_CONTIGUOUS(self.ptr()) != 0;
        }

        /// Check if array is contiguous (C-style)
        pub fn isContiguous(self: Self) bool {
            return numpy_ffi.pyz3_PyArray_ISCONTIGUOUS(self.ptr()) != 0;
        }

        /// Check if array is writeable
        pub fn isWriteable(self: Self) bool {
            return numpy_ffi.pyz3_PyArray_ISWRITEABLE(self.ptr()) != 0;
        }

        /// Check if array is aligned
        pub fn isAligned(self: Self) bool {
            return numpy_ffi.pyz3_PyArray_ISALIGNED(self.ptr()) != 0;
        }

        // ====================================================================
        // Zero-copy slice access
        // ====================================================================

        /// Get array data as a typed slice (zero-copy, read-only)
        /// WARNING: The returned slice is only valid while the array is alive!
        pub fn asSlice(self: Self, comptime T: type) PyError![]const T {
            // Verify dtype matches
            const expected = DType.fromType(T);
            const actual = self.dtype() orelse {
                return py.TypeError(root).raise("Array has unsupported dtype");
            };

            if (expected != actual) {
                return py.TypeError(root).raise("Type mismatch: array dtype does not match requested type");
            }

            // Must be C-contiguous for safe slice access
            if (!self.isCContiguous()) {
                return py.ValueError(root).raise("Array must be C-contiguous for slice access");
            }

            const ptr_data: [*]const T = self.dataAs(T) orelse {
                return py.ValueError(root).raise("Array has null or misaligned data pointer");
            };
            const len = self.size();
            return ptr_data[0..len];
        }

        /// Get array data as a mutable typed slice (zero-copy)
        /// WARNING: The returned slice is only valid while the array is alive!
        pub fn asSliceMut(self: Self, comptime T: type) PyError![]T {
            if (!self.isWriteable()) {
                return py.ValueError(root).raise("Array is not writeable");
            }

            // Verify dtype matches
            const expected = DType.fromType(T);
            const actual = self.dtype() orelse {
                return py.TypeError(root).raise("Array has unsupported dtype");
            };

            if (expected != actual) {
                return py.TypeError(root).raise("Type mismatch: array dtype does not match requested type");
            }

            // Must be C-contiguous for safe slice access
            if (!self.isCContiguous()) {
                return py.ValueError(root).raise("Array must be C-contiguous for slice access");
            }

            const ptr_data: [*]T = self.dataAs(T) orelse {
                return py.ValueError(root).raise("Array has null or misaligned data pointer");
            };
            const len = self.size();
            return ptr_data[0..len];
        }

        // ====================================================================
        // Array creation via Python calls (uses numpy module)
        // ====================================================================

        /// Create a new array from a Zig slice (copies data)
        pub fn fromSlice(comptime T: type, slice_data: []const T) PyError!Self {
            const np = py.import(root, "numpy") catch return PyError.PyRaised;
            defer np.decref();

            const array_func = np.get("array") catch return PyError.PyRaised;
            defer array_func.decref();

            // Create Python list from Zig slice
            const list = py.PyList(root).new(0) catch return PyError.PyRaised;
            defer list.obj.decref();

            for (slice_data) |item| {
                list.append(item) catch return PyError.PyRaised;
            }

            // Get dtype name
            const dtype_name = DType.fromType(T).name();

            // Call numpy.array(list, dtype=dtype_name)
            const result = py.call(root, py.PyObject, array_func, .{list.obj}, .{ .dtype = dtype_name }) catch return PyError.PyRaised;
            return Self{ .obj = result };
        }

        /// Create a zeros array
        pub fn zeros(comptime T: type, arr_shape: []const usize) PyError!Self {
            const np = py.import(root, "numpy") catch return PyError.PyRaised;
            defer np.decref();

            const zeros_func = np.get("zeros") catch return PyError.PyRaised;
            defer zeros_func.decref();

            // Create shape tuple
            const shape_tuple = shapeToPyTuple(arr_shape) catch return PyError.PyRaised;
            defer shape_tuple.decref();

            const dtype_name = DType.fromType(T).name();
            const result = py.call(root, py.PyObject, zeros_func, .{shape_tuple}, .{ .dtype = dtype_name }) catch return PyError.PyRaised;
            return Self{ .obj = result };
        }

        /// Create a ones array
        pub fn ones(comptime T: type, arr_shape: []const usize) PyError!Self {
            const np = py.import(root, "numpy") catch return PyError.PyRaised;
            defer np.decref();

            const ones_func = np.get("ones") catch return PyError.PyRaised;
            defer ones_func.decref();

            const shape_tuple = shapeToPyTuple(arr_shape) catch return PyError.PyRaised;
            defer shape_tuple.decref();

            const dtype_name = DType.fromType(T).name();
            const result = py.call(root, py.PyObject, ones_func, .{shape_tuple}, .{ .dtype = dtype_name }) catch return PyError.PyRaised;
            return Self{ .obj = result };
        }

        /// Create an empty array
        pub fn empty(comptime T: type, arr_shape: []const usize) PyError!Self {
            const np = py.import(root, "numpy") catch return PyError.PyRaised;
            defer np.decref();

            const empty_func = np.get("empty") catch return PyError.PyRaised;
            defer empty_func.decref();

            const shape_tuple = shapeToPyTuple(arr_shape) catch return PyError.PyRaised;
            defer shape_tuple.decref();

            const dtype_name = DType.fromType(T).name();
            const result = py.call(root, py.PyObject, empty_func, .{shape_tuple}, .{ .dtype = dtype_name }) catch return PyError.PyRaised;
            return Self{ .obj = result };
        }

        // ====================================================================
        // Array operations via Python calls
        // ====================================================================

        /// Get sum of all elements
        pub fn sum(self: Self, comptime T: type) PyError!T {
            const sum_method = self.obj.get("sum") catch return PyError.PyRaised;
            defer sum_method.decref();

            const result = py.call0(root, py.PyObject, sum_method) catch return PyError.PyRaised;
            defer result.decref();

            return py.as(root, T, result) catch return PyError.PyRaised;
        }

        /// Get mean of all elements
        pub fn mean(self: Self) PyError!f64 {
            const mean_method = self.obj.get("mean") catch return PyError.PyRaised;
            defer mean_method.decref();

            const result = py.call0(root, py.PyObject, mean_method) catch return PyError.PyRaised;
            defer result.decref();

            return py.as(root, f64, result) catch return PyError.PyRaised;
        }

        /// Get minimum value
        pub fn min(self: Self, comptime T: type) PyError!T {
            const min_method = self.obj.get("min") catch return PyError.PyRaised;
            defer min_method.decref();

            const result = py.call0(root, py.PyObject, min_method) catch return PyError.PyRaised;
            defer result.decref();

            return py.as(root, T, result) catch return PyError.PyRaised;
        }

        /// Get maximum value
        pub fn max(self: Self, comptime T: type) PyError!T {
            const max_method = self.obj.get("max") catch return PyError.PyRaised;
            defer max_method.decref();

            const result = py.call0(root, py.PyObject, max_method) catch return PyError.PyRaised;
            defer result.decref();

            return py.as(root, T, result) catch return PyError.PyRaised;
        }

        /// Reshape the array (returns new array)
        pub fn reshape(self: Self, new_shape: []const usize) PyError!Self {
            const reshape_method = self.obj.get("reshape") catch return PyError.PyRaised;
            defer reshape_method.decref();

            const shape_tuple = shapeToPyTuple(new_shape) catch return PyError.PyRaised;
            defer shape_tuple.decref();

            const result = py.call(root, py.PyObject, reshape_method, .{shape_tuple}, null) catch return PyError.PyRaised;
            return Self{ .obj = result };
        }

        /// Flatten the array to 1D
        pub fn flatten(self: Self) PyError!Self {
            const flatten_method = self.obj.get("flatten") catch return PyError.PyRaised;
            defer flatten_method.decref();

            const result = py.call0(root, py.PyObject, flatten_method) catch return PyError.PyRaised;
            return Self{ .obj = result };
        }

        /// Copy the array
        pub fn copy(self: Self) PyError!Self {
            const copy_method = self.obj.get("copy") catch return PyError.PyRaised;
            defer copy_method.decref();

            const result = py.call0(root, py.PyObject, copy_method) catch return PyError.PyRaised;
            return Self{ .obj = result };
        }

        /// Transpose the array
        pub fn transpose(self: Self) PyError!Self {
            const t_attr = self.obj.get("T") catch return PyError.PyRaised;
            return Self{ .obj = t_attr };
        }
    };
}

/// Convert a Zig usize slice to a Python tuple
fn shapeToPyTuple(arr_shape: []const usize) PyError!py.PyObject {
    const tuple = ffi.PyTuple_New(@intCast(arr_shape.len)) orelse return PyError.PyRaised;

    for (arr_shape, 0..) |dim, i| {
        const dim_obj = ffi.PyLong_FromUnsignedLongLong(dim) orelse {
            ffi.Py_DECREF(tuple);
            return PyError.PyRaised;
        };
        // PyTuple_SetItem steals the reference
        if (ffi.PyTuple_SetItem(tuple, @intCast(i), dim_obj) < 0) {
            ffi.Py_DECREF(tuple);
            return PyError.PyRaised;
        }
    }

    return py.PyObject{ .py = tuple };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "DType.fromType" {
    try testing.expectEqual(DType.int32, DType.fromType(i32));
    try testing.expectEqual(DType.float64, DType.fromType(f64));
    try testing.expectEqual(DType.uint8, DType.fromType(u8));
    try testing.expectEqual(DType.int64, DType.fromType(i64));
}

test "DType.size" {
    try testing.expectEqual(@as(usize, 1), DType.int8.size());
    try testing.expectEqual(@as(usize, 4), DType.int32.size());
    try testing.expectEqual(@as(usize, 8), DType.float64.size());
}

test "DType.name" {
    try testing.expectEqualStrings("int32", DType.int32.name());
    try testing.expectEqualStrings("float64", DType.float64.name());
    try testing.expectEqualStrings("bool", DType.bool_.name());
}
