const std = @import("std");
const py = @import("../pyz3.zig");
const PyObjectMixin = @import("./obj.zig").PyObjectMixin;
const ffi = py.ffi;
const PyObject = @import("obj.zig").PyObject;
const PyError = @import("../errors.zig").PyError;
const State = @import("../discovery.zig").State;

pub const PyString = extern struct {
    obj: PyObject,

    const Self = @This();
    pub const from = PyObjectMixin("str", "PyUnicode", Self);

    pub fn create(value: []const u8) !Self {
        const unicode = ffi.PyUnicode_FromStringAndSize(value.ptr, @intCast(value.len)) orelse return PyError.PyRaised;
        return .{ .obj = .{ .py = unicode } };
    }

    pub fn createFmt(comptime format: []const u8, args: anytype) !Self {
        const str = try std.fmt.allocPrint(py.allocator, format, args);
        defer py.allocator.free(str);
        return create(str);
    }

    /// Append other to self.
    ///
    /// Warning: a reference to self is stolen. Use concat, or self.obj.incref(), if you don't own a reference to self.
    pub fn append(self: Self, other: Self) !Self {
        return self.appendObj(other.obj);
    }

    /// Append the slice to self.
    ///
    /// Warning: a reference to self is stolen. Use concat, or self.obj.incref(), if you don't own a reference to self.
    pub fn appendSlice(self: Self, str: []const u8) !Self {
        const other = try create(str);
        defer other.obj.decref();
        return self.appendObj(other.obj);
    }

    fn appendObj(self: Self, other: PyObject) !Self {
        // Note: PyUnicode_Append modifies the first argument in-place and decrefs it on error.
        // This is different from typical Python semantics where strings are immutable.
        // We expose this behavior in the API since it matches the underlying CPython function.
        // The caller is responsible for managing the reference count appropriately.
        var self_ptr: ?*ffi.PyObject = self.obj.py;
        ffi.PyUnicode_Append(&self_ptr, other.py);
        if (self_ptr) |ptr| {
            return Self.from.unchecked(.{ .py = ptr });
        } else {
            // If set to null, then it failed.
            return PyError.PyRaised;
        }
    }

    /// Concat other to self. Returns a new reference.
    pub fn concat(self: Self, other: Self) !Self {
        const result = ffi.PyUnicode_Concat(self.obj.py, other.obj.py) orelse return PyError.PyRaised;
        return Self.from.unchecked(.{ .py = result });
    }

    /// Concat other to self. Returns a new reference.
    pub fn concatSlice(self: Self, other: []const u8) !Self {
        const otherString = try create(other);
        defer otherString.obj.decref();

        return concat(self, otherString);
    }

    /// Return the length of the Unicode object, in code points.
    pub fn length(self: Self) !usize {
        return @intCast(ffi.PyUnicode_GetLength(self.obj.py));
    }

    /// Returns a view over the PyString bytes.
    pub fn asSlice(self: Self) ![:0]const u8 {
        var size: i64 = 0;
        const buffer: [*:0]const u8 = ffi.PyUnicode_AsUTF8AndSize(self.obj.py, &size) orelse return PyError.PyRaised;
        return buffer[0..@as(usize, @intCast(size)) :0];
    }
};

const testing = std.testing;

test "PyString" {
    py.initialize();
    defer py.finalize();

    const a = "Hello";
    const b = ", world!";

    var ps = try PyString.create(a);
    // defer ps.decref();  <-- We don't need to decref here since append steals the reference to self.
    ps = try ps.appendSlice(b);
    defer ps.obj.decref();

    const ps_slice = try ps.asSlice();

    // Null-terminated strings have len == non-null bytes, but are guaranteed to have a null byte
    // when indexed by their length.
    try testing.expectEqual(a.len + b.len, ps_slice.len);
    try testing.expectEqual(@as(u8, 0), ps_slice[ps_slice.len]);

    try testing.expectEqualStrings("Hello, world!", ps_slice);
}

test "PyString createFmt" {
    py.initialize();
    defer py.finalize();

    const a = try PyString.createFmt("Hello, {s}!", .{"foo"});
    defer a.obj.decref();

    try testing.expectEqualStrings("Hello, foo!", try a.asSlice());
}
