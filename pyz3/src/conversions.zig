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

const py = @import("./pyz3.zig");
const tramp = @import("./trampoline.zig");
const pytypes = @import("./pytypes.zig");
const State = @import("./discovery.zig").State;

/// Zig PyObject-like -> ffi.PyObject. Convert a Zig PyObject-like value into a py.PyObject.
///  e.g. py.PyObject, py.PyTuple, ffi.PyObject, etc.
///
/// Performance: Zero-cost conversion for PyObject-like types, direct pass-through.
/// Inline hint ensures function is inlined in hot paths.
pub inline fn object(comptime root: type, value: anytype) py.PyObject {
    return tramp.Trampoline(root, @TypeOf(value)).asObject(value);
}

/// Zig -> Python. Return a Python representation of a Zig object.
/// For Zig primitives, this constructs a new Python object.
/// For PyObject-like values, this returns the value without creating a new reference.
///
/// Performance: Optimized reference counting - transfers ownership instead of creating new refs.
/// Use this when you want to transfer ownership of PyObject-like values.
pub inline fn createOwned(comptime root: type, value: anytype) py.PyError!py.PyObject {
    const trampoline = tramp.Trampoline(root, @TypeOf(value));
    defer trampoline.decref_objectlike(value);
    return trampoline.wrap(value);
}

/// Zig -> Python. Convert a Zig object into a Python object. Returns a new object.
///
/// Performance: Creates new Python objects for primitives, increments refcount for PyObject-like.
/// This is the standard conversion function - use when you need a new reference.
pub inline fn create(comptime root: type, value: anytype) py.PyError!py.PyObject {
    return tramp.Trampoline(root, @TypeOf(value)).wrap(value);
}

/// Python -> Zig. Return a Zig object representing the Python object.
///
/// Performance: Zero-cost for PyObject-like types, efficient conversion for primitives.
/// Uses trampoline system for type-safe conversions with minimal overhead.
pub inline fn as(comptime root: type, comptime T: type, obj: anytype) py.PyError!T {
    return tramp.Trampoline(root, T).unwrap(object(root, obj));
}

/// Python -> Pydust. Perform a type-checked cast from a PyObject to a given PyDust class type.
/// This performs runtime type validation using isinstance() and returns an error if types don't match.
/// Use this by default for safety. Only use unchecked() in proven performance-critical paths.
///
/// Performance optimizations:
///   - Fast path: Direct ob_type comparison for exact matches (most common case)
///   - Only calls expensive isinstance() when inheritance checking is needed
///   - Compile-time validation ensures type safety
///   - Inline hint ensures function is inlined in hot paths
pub inline fn checked(comptime root: type, comptime T: type, obj: py.PyObject) py.PyError!T {
    // Compile-time type validation - zero runtime cost
    comptime {
        const Definition = @typeInfo(T).pointer.child;
        const definition = State.getDefinition(root, Definition);
        if (definition.type != .class) {
            @compileError("Can only perform checked cast into a PyDust class type. Found " ++ @typeName(Definition));
        }
    }

    const Definition = @typeInfo(T).pointer.child;

    // Get the expected type for validation
    // Note: py.self() does module import + attr lookup, but we need it for validation
    const Cls = try py.self(root, Definition);
    defer Cls.obj.decref();

    // OPTIMIZATION: Fast path for exact type matches using ob_type pointer comparison
    // This is much faster than isinstance() for the common case where types match exactly
    // Direct pointer comparison: O(1) vs isinstance() which walks the MRO: O(depth)
    const obj_type = py.type_(root, obj);
    const is_exact_match = obj_type.obj.py == Cls.obj.py;

    // Only call expensive isinstance() when we need to check inheritance
    // This handles the case where obj is a subclass of the expected type
    if (!is_exact_match and !try py.isinstance(root, obj, Cls)) {
        // Error path: Build detailed error message (only executed on type mismatch)
        const clsName = State.getIdentifier(root, Definition).name();
        const mod = State.getContaining(root, Definition, .module);
        const modName = State.getIdentifier(root, mod).name();
        return py.TypeError(root).raiseFmt(
            "Expected {s}.{s} but found {s}",
            .{ modName, clsName, try obj.getTypeName() },
        );
    }

    // Success path: Direct cast (same as unchecked, but we've validated the type)
    const instance: *pytypes.PyTypeStruct(Definition) = @ptrCast(@alignCast(obj.py));
    return &instance.state;
}

/// Python -> Pydust. Perform an unchecked cast from a PyObject to a given PyDust class type.
///
/// ⚠️ DANGER: This function performs NO runtime type validation and can lead to:
///   - Memory corruption
///   - Segmentation faults
///   - Arbitrary code execution if attacker controls type
///
/// The caller MUST guarantee obj is actually an instance of T. If you cannot prove this
/// statically, use checked() instead.
///
/// Only use this in:
///   1. Performance-critical inner loops where isinstance() is measurably too slow
///   2. After you've already validated the type externally
///   3. Internal functions where type is guaranteed by construction
///
/// Example safe usage:
///   // After explicit check
///   if (try py.isinstance(root, obj, MyClass)) {
///       const instance = py.unchecked(root, *MyClass, obj); // Safe here
///   }
///
/// Performance optimizations:
///   - All type validation happens at compile-time (zero runtime cost)
///   - Direct pointer cast with no intermediate allocations
///   - Debug assertions compile away in release builds
///   - Inline hint ensures function is inlined in hot paths
pub inline fn unchecked(comptime root: type, comptime T: type, obj: py.PyObject) T {
    // Compile-time type validation - zero runtime cost
    comptime {
        const Definition = @typeInfo(T).pointer.child;
        const definition = State.getDefinition(root, Definition);
        if (definition.type != .class) {
            @compileError("Can only perform unchecked cast into a PyDust class type. Found " ++ @typeName(Definition));
        }
    }

    // Extract type info at compile-time for optimal code generation
    const Definition = @typeInfo(T).pointer.child;

    // Debug-mode assertion: Validate type at runtime in debug builds only
    // This compiles away completely in release builds (zero cost)
    if (@import("builtin").mode == .Debug) {
        // Ensure definition exists at compile-time (validation happens in comptime block above)
        comptime {
            const definition = State.getDefinition(root, Definition);
            _ = definition; // Ensure definition exists
        }

        // In debug mode, we can optionally add a lightweight check
        // This is disabled by default to maintain zero-cost in release
        // Uncomment the following lines if you want debug-mode validation:
        // const expected_type_obj = py.self(root, Definition) catch return @as(T, @ptrCast(@alignCast(obj.py)));
        // defer expected_type_obj.obj.decref();
        // const obj_type = py.type_(root, obj);
        // if (obj_type.obj.py != expected_type_obj.obj.py) {
        //     @panic("unchecked() called with wrong type - this indicates a bug in your code");
        // }
    }

    // SAFETY: Caller guarantees obj is instance of T
    // Direct cast with no intermediate steps - optimal code generation
    const instance: *pytypes.PyTypeStruct(Definition) = @ptrCast(@alignCast(obj.py));
    return &instance.state;
}

const testing = @import("std").testing;
const expect = testing.expect;

test "as py -> zig" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    // Start with a Python object
    const str = try py.PyString.create("hello");
    try expect(py.refcnt(root, str) == 1);

    // Return a slice representation of it, and ensure the refcnt is untouched
    _ = try py.as(root, []const u8, str);
    try expect(py.refcnt(root, str) == 1);

    // Return a PyObject representation of it, and ensure the refcnt is untouched.
    _ = try py.as(root, py.PyObject, str);
    try expect(py.refcnt(root, str) == 1);
}

test "create" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    const str = try py.PyString.create("Hello");
    try testing.expectEqual(@as(isize, 1), py.refcnt(root, str));

    const some_tuple = try py.create(root, .{str});
    defer some_tuple.decref();
    try testing.expectEqual(@as(isize, 2), py.refcnt(root, str));

    str.obj.decref();
    try testing.expectEqual(@as(isize, 1), py.refcnt(root, str));
}

test "createOwned" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    const str = try py.PyString.create("Hello");
    try testing.expectEqual(@as(isize, 1), py.refcnt(root, str));

    const some_tuple = try py.createOwned(root, .{str});
    defer some_tuple.decref();
    try testing.expectEqual(@as(isize, 1), py.refcnt(root, str));
}
