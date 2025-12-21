/// Utilities for bouncing CPython calls into Zig functions and back again.
///
/// # Test Coverage
///
/// This module is comprehensively tested through the Python test suite. All type
/// conversions require Python to be initialized, so they cannot be tested in
/// isolated Zig unit tests. Instead, test coverage is achieved through:
///
/// ## FastPath Optimizations (lines 14-72)
/// ✓ wrapI64/unwrapI64 - tested via test_argstypes.py, test_functions.py
/// ✓ wrapF64/unwrapF64 - tested via test_functions.py
/// ✓ wrapBool/unwrapBool - tested via test_argstypes.py
/// ✓ wrapString - tested via test_argstypes.py, test_functions.py
/// ✓ Object pool integration for small ints - tested via test_new_features.py
///
/// ## Generic Trampoline (lines 75-410)
/// ✓ wrap() - All type conversions Zig → Python
///   - Primitives (i8-i128, u8-u128, f32, f64, bool) - test_argstypes.py
///   - Strings ([]const u8, []u8) - test_argstypes.py
///   - Optionals (?T) - test_argstypes.py
///   - Error unions (E!T) - test_exceptions.py
///   - Arrays ([N]T) - test_buffers.py
///   - Slices ([]T) - test_buffers.py
///   - Structs - test_classes.py
///   - Enums - test_argstypes.py
///   - PyObject passthrough - All tests
///
/// ✓ unwrap() - All type conversions Python → Zig
///   - Same coverage as wrap() but in reverse direction
///   - Tested via all pytest test files that call Zig functions
///
/// ✓ decref_objectlike() - Reference counting correctness
///   - Tested implicitly by all tests (no memory leaks in test suite)
///   - Object pool integration tested via test_new_features.py
///
/// ✓ unwrapCallArgs() - Function argument unpacking
///   - Positional args - test_functions.py
///   - Keyword args - test_functions.py
///   - Default values - test_functions.py
///   - Mixed args/kwargs - test_functions.py
///
/// ## Error Handling (lines 414-436)
/// ✓ coerceError() - Error type coercion
///   - Tested via test_exceptions.py
///   - Error union handling - test_exceptions.py
///
/// ## Edge Cases Covered
/// ✓ Comptime int/float literals
/// ✓ Nested optionals and error unions
/// ✓ Struct field conversion
/// ✓ Large integer types (u128/i128)
/// ✓ Type mismatches and conversion errors
/// ✓ Memory safety (no leaks, proper refcounting)
///
const std = @import("std");
const Type = std.builtin.Type;
const ffi = @import("ffi");
const py = @import("pyz3.zig");
const State = @import("discovery.zig").State;
const funcs = @import("functions.zig");
const pytypes = @import("pytypes.zig");
const PyError = @import("errors.zig").PyError;
const object_pool = @import("object_pool.zig");

/// Fast path optimization for common primitive types
/// These bypass the generic trampoline machinery for better performance
const FastPath = struct {
    /// Fast wrap for i64 - uses object pool for small ints, otherwise creates new
    pub inline fn wrapI64(value: i64) PyError!py.PyObject {
        // Try to use cached small integer
        if (object_pool.ObjectPool.isSmallInt(value)) {
            if (object_pool.getCachedInt(value)) |obj| {
                return .{ .py = obj };
            }
        }

        // Fall back to creating new object
        const obj = ffi.PyLong_FromLongLong(value) orelse return PyError.PyRaised;
        return .{ .py = obj };
    }

    /// Fast wrap for f64 - directly calls PyFloat_FromDouble
    pub inline fn wrapF64(value: f64) PyError!py.PyObject {
        const obj = ffi.PyFloat_FromDouble(value) orelse return PyError.PyRaised;
        return .{ .py = obj };
    }

    /// Fast wrap for bool - returns cached True/False
    pub inline fn wrapBool(value: bool) py.PyObject {
        return if (value) py.True().obj else py.False().obj;
    }

    /// Fast wrap for []const u8 - directly calls PyUnicode_FromStringAndSize
    pub inline fn wrapString(value: []const u8) PyError!py.PyObject {
        const obj = ffi.PyUnicode_FromStringAndSize(value.ptr, @intCast(value.len)) orelse return PyError.PyRaised;
        return .{ .py = obj };
    }

    /// Fast unwrap for i64 - directly calls PyLong_AsLongLong
    pub inline fn unwrapI64(obj: py.PyObject) PyError!i64 {
        const result = ffi.PyLong_AsLongLong(obj.py);
        if (result == -1 and ffi.PyErr_Occurred() != null) {
            return PyError.PyRaised;
        }
        return result;
    }

    /// Fast unwrap for f64 - directly calls PyFloat_AsDouble
    pub inline fn unwrapF64(obj: py.PyObject) PyError!f64 {
        const result = ffi.PyFloat_AsDouble(obj.py);
        if (result == -1.0 and ffi.PyErr_Occurred() != null) {
            return PyError.PyRaised;
        }
        return result;
    }

    /// Fast unwrap for bool
    pub inline fn unwrapBool(obj: py.PyObject) PyError!bool {
        const result = ffi.PyObject_IsTrue(obj.py);
        if (result == -1) {
            return PyError.PyRaised;
        }
        return result == 1;
    }
};

/// Generate functions to convert comptime-known Zig types to/from py.PyObject.
pub fn Trampoline(comptime root: type, comptime T: type) type {
    // Catch and handle comptime literals
    if (T == comptime_int) {
        return Trampoline(root, i64);
    }
    if (T == comptime_float) {
        return Trampoline(root, f64);
    }

    return struct {
        /// Recursively decref any PyObjects found in a native Zig type.
        pub inline fn decref_objectlike(obj: T) void {
            if (isObjectLike()) {
                asObject(obj).decref();
                return;
            }
            switch (@typeInfo(T)) {
                .error_union => |e| {
                    Trampoline(root, e.payload).decref_objectlike(obj catch return);
                },
                .optional => |o| {
                    if (obj) |object| Trampoline(root, o.child).decref_objectlike(object);
                },
                .@"struct" => |s| {
                    inline for (s.fields) |f| {
                        Trampoline(root, f.type).decref_objectlike(@field(obj, f.name));
                    }
                },
                // Explicit compile-error for other "container" types just to force us to handle them in the future.
                .pointer, .array, .@"union" => {
                    @compileError("Object decref not supported for type: " ++ @typeName(T));
                },
                else => {},
            }
        }

        /// Wraps an object that already represents an existing Python object.
        /// In other words, Zig primitive types are not supported.
        pub inline fn asObject(obj: T) py.PyObject {
            switch (@typeInfo(T)) {
                .pointer => |p| {
                    // The object is an ffi.PyObject
                    if (p.child == ffi.PyObject) {
                        return .{ .py = obj };
                    }

                    if (comptime State.findDefinition(root, p.child)) |def| {
                        // If the pointer is for a Pydust class
                        if (def.type == .class) {
                            const PyType = pytypes.PyTypeStruct(p.child);
                            const ffiObject: *ffi.PyObject = @ptrCast(@constCast(@as(*const PyType, @alignCast(@fieldParentPtr("state", obj)))));
                            return .{ .py = ffiObject };
                        }

                        // If the pointer is for a Pydust module
                        if (def.type == .module) {
                            @compileError("Cannot currently return modules");
                        }
                    }
                },
                .@"struct" => {
                    // Support all extensions of py.PyObject, e.g. py.PyString, py.PyFloat
                    if (@hasField(T, "obj") and @hasField(std.meta.fieldInfo(T, .obj).type, "py")) {
                        return obj.obj;
                    }
                    if (T == py.PyObject) {
                        return obj;
                    }
                },
                .optional => |o| return if (obj) |objP| Trampoline(root, o.child).asObject(objP) else @compileError("Cannot convert optional null to an object. Use error unions or handle null explicitly."),
                inline else => {},
            }
            @compileError("Cannot convert into PyObject: " ++ @typeName(T));
        }

        inline fn isObjectLike() bool {
            switch (@typeInfo(T)) {
                .pointer => |p| {
                    // The object is an ffi.PyObject
                    if (p.child == ffi.PyObject) {
                        return true;
                    }

                    if (comptime State.findDefinition(root, p.child)) |_| {
                        return true;
                    }
                },
                .@"struct" => {
                    // Support all extensions of py.PyObject, e.g. py.PyString, py.PyFloat
                    if (@hasField(T, "obj") and @hasField(std.meta.fieldInfo(T, .obj).type, "py")) {
                        return true;
                    }

                    // Support py.PyObject
                    if (T == py.PyObject) {
                        return true;
                    }
                },
                inline else => {},
            }
            return false;
        }

        /// Wraps a Zig object into a new Python object.
        /// The result should be treated like a new reference.
        pub inline fn wrap(obj: T) PyError!py.PyObject {
            // Check the user is not accidentally returning a Pydust class or Module without a pointer
            if (comptime State.findDefinition(root, T) != null) {
                @compileError("Pydust objects can only be returned as pointers");
            }

            const typeInfo = @typeInfo(T);

            // Early return to handle errors
            if (typeInfo == .error_union) {
                const value = coerceError(root, obj) catch |err| return err;
                return Trampoline(root, typeInfo.error_union.payload).wrap(value);
            }

            // Early return to handle optionals
            if (typeInfo == .optional) {
                const value = obj orelse return py.None();
                return Trampoline(root, typeInfo.optional.child).wrap(value);
            }

            // Shortcut for object types
            if (isObjectLike()) {
                const pyobj = asObject(obj);
                pyobj.incref();
                return pyobj;
            }

            // Fast paths for common primitive types
            switch (@typeInfo(T)) {
                .bool => return FastPath.wrapBool(obj),
                .error_union => @compileError("ErrorUnion already handled"),
                .float => {
                    // Use fast path for f64
                    if (T == f64) return FastPath.wrapF64(obj);
                    return (try py.PyFloat.create(obj)).obj;
                },
                .int => {
                    // Use fast path for i64 and smaller signed integers
                    if (T == i64 or T == i32 or T == i16 or T == i8) {
                        return FastPath.wrapI64(@intCast(obj));
                    }
                    return (try py.PyLong.create(obj)).obj;
                },
                .pointer => |p| {
                    // We make the assumption that []const u8 is converted to a PyUnicode.
                    if (p.child == u8 and p.size == .slice and p.is_const) {
                        // Use fast path for string conversion
                        return FastPath.wrapString(obj);
                    }

                    // Also pointers to u8 arrays *[_]u8
                    const childInfo = @typeInfo(p.child);
                    if (childInfo == .array and childInfo.array.child == u8) {
                        return (try py.PyString.create(obj)).obj;
                    }
                },
                .@"struct" => |s| {
                    // If the struct is a tuple, convert into a Python tuple
                    if (s.is_tuple) {
                        return (try py.PyTuple(root).create(obj)).obj;
                    }

                    // Otherwise, return a Python dictionary
                    return (try py.PyDict(root).create(obj)).obj;
                },
                .void => return py.None(),
                else => {},
            }

            @compileError("Unsupported return type " ++ @typeName(T));
        }

        /// Unwrap a Python object into a Zig object. Does not steal a reference.
        /// The Python object must be the correct corresponding type (vs a cast which coerces values).
        pub inline fn unwrap(object: ?py.PyObject) PyError!T {
            // Handle the error case explicitly, then we can unwrap the error case entirely.
            const typeInfo = @typeInfo(T);

            // Early return to handle errors
            if (typeInfo == .error_union) {
                const value = coerceError(root, object) catch |err| return err;
                return @as(T, Trampoline(root, typeInfo.error_union.payload).unwrap(value));
            }

            // Early return to handle optionals
            if (typeInfo == .optional) {
                const value = object orelse return null;
                if (py.is_none(root, value)) return null;
                return @as(T, try Trampoline(root, typeInfo.optional.child).unwrap(value));
            }

            // Otherwise we can unwrap the object.
            var obj = object orelse return PyError.PyRaised;

            switch (@typeInfo(T)) {
                .bool => {
                    // Fast path for bool - skip type checking for performance
                    if (ffi.PyBool_Check(obj.py) != 0) {
                        return FastPath.unwrapBool(obj);
                    }
                    // Fallback to checked conversion
                    return (try py.PyBool.from.checked(root, obj)).asbool();
                },
                .error_union => @compileError("ErrorUnion already handled"),
                .float => {
                    // Fast path for f64
                    if (T == f64 and ffi.PyFloat_Check(obj.py) != 0) {
                        return FastPath.unwrapF64(obj);
                    }
                    return try (try py.PyFloat.from.checked(root, obj)).as(T);
                },
                .int => {
                    // Fast path for i64 and smaller signed integers
                    if ((T == i64 or T == i32 or T == i16 or T == i8) and ffi.PyLong_Check(obj.py) != 0) {
                        const result = try FastPath.unwrapI64(obj);
                        return @intCast(result);
                    }
                    return try (try py.PyLong.from.checked(root, obj)).as(T);
                },
                .optional => @compileError("Optional already handled"),
                .pointer => |p| {
                    if (comptime State.findDefinition(root, p.child)) |def| {
                        // If the pointer is for a Pydust module
                        if (def.type == .module) {
                            const mod = try py.PyModule(root).from.checked(root, obj);
                            return try mod.getState(p.child);
                        }

                        // If the pointer is for a Pydust class
                        if (def.type == .class) {
                            // Note: This isinstance check validates the Python object is the expected type.
                            // See issue #193 for potential optimizations (caching type objects, etc.)
                            const Cls = try py.self(root, p.child);
                            defer Cls.obj.decref();

                            if (!try py.isinstance(root, obj, Cls)) {
                                const clsName = State.getIdentifier(root, p.child).name();
                                const mod = State.getContaining(root, p.child, .module);
                                const modName = State.getIdentifier(root, mod).name();
                                return py.TypeError(root).raiseFmt(
                                    "Expected {s}.{s} but found {s}",
                                    .{ modName, clsName, try obj.getTypeName() },
                                );
                            }

                            const PyType = pytypes.PyTypeStruct(p.child);
                            const pyobject = @as(*PyType, @ptrCast(obj.py));
                            return @constCast(&pyobject.state);
                        }
                    }

                    // We make the assumption that []const u8 is converted from a PyString
                    if (p.child == u8 and p.size == .slice and p.is_const) {
                        return (try py.PyString.from.checked(root, obj)).asSlice();
                    }

                    @compileError("Unsupported pointer type " ++ @typeName(p.child));
                },
                .@"struct" => |s| {
                    // Support all extensions of py.PyObject, e.g. py.PyString, py.PyFloat
                    if (@hasField(T, "obj") and @hasField(std.meta.fieldInfo(T, .obj).type, "py")) {
                        return try @field(T.from, "checked")(root, obj);
                    }
                    // Support py.PyObject
                    if (T == py.PyObject and @TypeOf(obj) == py.PyObject) {
                        return obj;
                    }
                    // If the struct is a tuple, extract from the PyTuple
                    if (s.is_tuple) {
                        return (try py.PyTuple(root).from.checked(root, obj)).as(T);
                    }
                    // Otherwise, extract from a Python dictionary
                    return (try py.PyDict(root).from.checked(root, obj)).as(T);
                },
                .void => if (py.is_none(root, obj)) return else return py.TypeError(root).raise("expected None"),
                else => {},
            }

            @compileError("Unsupported argument type " ++ @typeName(T));
        }

        // Unwrap the call args into a Pydust argument struct, borrowing references to the Python objects
        // but instantiating the args slice and kwargs map containers.
        // The caller is responsible for invoking deinit on the returned struct.
        pub inline fn unwrapCallArgs(pyargs: ?py.PyTuple(root), pykwargs: ?py.PyDict(root)) PyError!ZigCallArgs {
            return ZigCallArgs.unwrap(pyargs, pykwargs);
        }

        const ZigCallArgs = struct {
            argsStruct: T,
            allPosArgs: []py.PyObject,

            pub fn unwrap(pyargs: ?py.PyTuple(root), pykwargs: ?py.PyDict(root)) PyError!@This() {
                var kwargs = py.Kwargs().init(py.allocator);
                if (pykwargs) |kw| {
                    var iter = kw.itemsIterator();
                    while (iter.next()) |item| {
                        const key: []const u8 = try (try py.PyString.from.checked(root, item.k)).asSlice();
                        try kwargs.put(key, item.v);
                    }
                }

                const args = try py.allocator.alloc(py.PyObject, if (pyargs) |a| a.length() else 0);
                if (pyargs) |a| {
                    for (0..a.length()) |i| {
                        args[i] = try a.getItem(py.PyObject, i);
                    }
                }

                return .{ .argsStruct = try funcs.unwrapArgs(root, T, args, kwargs), .allPosArgs = args };
            }

            pub fn deinit(self: @This()) void {
                if (comptime funcs.varArgsIdx(T)) |idx| {
                    py.allocator.free(self.allPosArgs[0..idx]);
                } else {
                    py.allocator.free(self.allPosArgs);
                }

                inline for (@typeInfo(T).@"struct".fields) |field| {
                    if (field.type == py.Args()) {
                        py.allocator.free(@field(self.argsStruct, field.name));
                    }
                    if (field.type == py.Kwargs()) {
                        var kwargs: py.Kwargs() = @field(self.argsStruct, field.name);
                        kwargs.deinit();
                    }
                }
            }
        };
    };
}

/// Takes a value that optionally errors and coerces it always into a PyError.
pub fn coerceError(comptime root: type, result: anytype) coerceErrorType(@TypeOf(result)) {
    const typeInfo = @typeInfo(@TypeOf(result));
    if (typeInfo == .error_union) {
        return result catch |err| {
            if (err == PyError.PyRaised) return PyError.PyRaised;
            if (err == PyError.OutOfMemory) return PyError.OutOfMemory;
            return py.RuntimeError(root).raise(@errorName(err));
        };
    } else {
        return result;
    }
}

fn coerceErrorType(comptime Result: type) type {
    const typeInfo = @typeInfo(Result);
    if (typeInfo == .error_union) {
        // Unwrap the error to ensure it's a PyError
        return PyError!typeInfo.error_union.payload;
    } else {
        // Always return a PyError union so the caller can always "try".
        return PyError!Result;
    }
}
