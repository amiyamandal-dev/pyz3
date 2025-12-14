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

const std = @import("std");
const py = @import("../pyz3.zig");
const PyObjectMixin = @import("./obj.zig").PyObjectMixin;
const ffi = py.ffi;
const PyObject = py.PyObject;
const PyLong = py.PyLong;
const PyError = @import("../errors.zig").PyError;
const State = @import("../discovery.zig").State;

/// Wrapper for Python PyList.
/// See: https://docs.python.org/3/c-api/list.html
pub fn PyList(comptime root: type) type {
    return extern struct {
        obj: PyObject,

        const Self = @This();
        pub const from = PyObjectMixin("list", "PyList", Self);

        pub fn new(size: usize) !Self {
            const list = ffi.PyList_New(@intCast(size)) orelse return PyError.PyRaised;
            return .{ .obj = .{ .py = list } };
        }

        pub fn length(self: Self) usize {
            return @intCast(ffi.PyList_Size(self.obj.py));
        }

        // Returns borrowed reference.
        pub fn getItem(self: Self, comptime T: type, idx: isize) !T {
            if (ffi.PyList_GetItem(self.obj.py, idx)) |item| {
                return py.as(root, T, py.PyObject{ .py = item });
            } else {
                return PyError.PyRaised;
            }
        }

        // Returns a slice of the list.
        pub fn getSlice(self: Self, low: isize, high: isize) !Self {
            if (ffi.PyList_GetSlice(self.obj.py, low, high)) |item| {
                return .{ .obj = .{ .py = item } };
            } else {
                return PyError.PyRaised;
            }
        }

        /// This function “steals” a reference to item and discards a reference to an item already in the list at the affected position.
        pub fn setOwnedItem(self: Self, pos: usize, value: anytype) !void {
            // Since this function steals the reference, it can only accept object-like values.
            if (ffi.PyList_SetItem(self.obj.py, @intCast(pos), py.object(root, value).py) < 0) {
                return PyError.PyRaised;
            }
        }

        /// Set the item at the given position.
        pub fn setItem(self: Self, pos: usize, value: anytype) !void {
            const valueObj = try py.create(root, value);
            return self.setOwnedItem(pos, valueObj);
        }

        // Insert the item item into list list in front of index idx.
        pub fn insert(self: Self, idx: isize, value: anytype) !void {
            const valueObj = try py.create(root, value);
            defer valueObj.decref();
            if (ffi.PyList_Insert(self.obj.py, idx, valueObj.py) < 0) {
                return PyError.PyRaised;
            }
        }

        // Append the object item at the end of list list.
        pub fn append(self: Self, value: anytype) !void {
            const valueObj = try py.create(root, value);
            defer valueObj.decref();

            if (ffi.PyList_Append(self.obj.py, valueObj.py) < 0) {
                return PyError.PyRaised;
            }
        }

        // Sort the items of list in place.
        pub fn sort(self: Self) !void {
            if (ffi.PyList_Sort(self.obj.py) < 0) {
                return PyError.PyRaised;
            }
        }

        // Reverse the items of list in place.
        pub fn reverse(self: Self) !void {
            if (ffi.PyList_Reverse(self.obj.py) < 0) {
                return PyError.PyRaised;
            }
        }

        pub fn toTuple(self: Self) !py.PyTuple(root) {
            const pytuple = ffi.PyList_AsTuple(self.obj.py) orelse return PyError.PyRaised;
            return py.PyTuple(root).from.unchecked(.{ .py = pytuple });
        }

        /// Create a PyList from a Zig slice or array
        /// Recursively handles nested collections via py.create()
        pub fn fromSlice(items: anytype) !Self {
            const T = @TypeOf(items);
            const typeInfo = @typeInfo(T);

            const slice = switch (typeInfo) {
                .pointer => |p| blk: {
                    if (p.size == .slice) {
                        break :blk items;
                    } else if (p.size == .one) {
                        // Handle pointer to array: *[N]T or *const [N]T
                        const childInfo = @typeInfo(p.child);
                        if (childInfo == .array) {
                            break :blk items[0..childInfo.array.len];
                        }
                    }
                    @compileError("Expected slice or pointer to array, got " ++ @typeName(T));
                },
                .array => &items,
                else => @compileError("Expected slice or array, got " ++ @typeName(T)),
            };

            const list = try new(slice.len);
            for (slice, 0..) |item, i| {
                // Recursively create each item (handles nested collections)
                try list.setOwnedItem(i, try py.create(root, item));
            }
            return list;
        }

        /// Convert PyList to Zig slice
        /// Caller owns returned memory allocated with py.allocator
        pub fn toSlice(self: Self, comptime T: type) ![]T {
            const len = self.length();
            const result = try py.allocator.alloc(T, len);
            errdefer py.allocator.free(result);

            for (0..len) |i| {
                // Recursively unwrap each element (handles nested collections)
                result[i] = try self.getItem(T, @intCast(i));
            }

            return result;
        }

        /// Create a PyList from a Zig slice/array value
        /// This is the high-level API matching PyDict.create() and PyTuple.create()
        pub fn create(value: anytype) !Self {
            const T = @TypeOf(value);
            const typeInfo = @typeInfo(T);

            // Validate that value is a slice, array, or pointer to array
            switch (typeInfo) {
                .pointer => |p| {
                    if (p.size == .slice) {
                        // Slice is OK
                    } else if (p.size == .one) {
                        const childInfo = @typeInfo(p.child);
                        if (childInfo != .array) {
                            @compileError("PyList.create expects a slice or array, got " ++ @typeName(T));
                        }
                    } else {
                        @compileError("PyList.create expects a slice or array, got " ++ @typeName(T));
                    }
                },
                .array => {},
                else => @compileError("PyList.create expects a slice or array, got " ++ @typeName(T)),
            }

            return fromSlice(value);
        }

        /// Convert PyList to Zig slice type
        /// Returns owned memory allocated with py.allocator - caller must free
        pub fn as(self: Self, comptime T: type) !T {
            const typeInfo = @typeInfo(T);

            // Validate T is a slice type
            if (typeInfo != .pointer) {
                @compileError("PyList.as expects a slice type, got " ++ @typeName(T));
            }
            if (typeInfo.pointer.size != .slice) {
                @compileError("PyList.as expects a slice type, got " ++ @typeName(T));
            }

            const ChildT = typeInfo.pointer.child;
            return self.toSlice(ChildT);
        }
    };
}

const testing = std.testing;

test "PyList" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    var list = try PyList(root).new(2);
    defer list.obj.decref();
    try list.setItem(0, 1);
    try list.setItem(1, 2.0);

    try testing.expectEqual(@as(usize, 2), list.length());

    try testing.expectEqual(@as(i64, 1), try list.getItem(i64, 0));
    try testing.expectEqual(@as(f64, 2.0), try list.getItem(f64, 1));

    try list.append(3);
    try testing.expectEqual(@as(usize, 3), list.length());
    try testing.expectEqual(@as(i32, 3), try list.getItem(i32, 2));

    try list.insert(0, 1.23);
    try list.reverse();
    try testing.expectEqual(@as(f32, 1.23), try list.getItem(f32, 3));

    try list.sort();
    try testing.expectEqual(@as(i64, 1), try list.getItem(i64, 0));

    const tuple = try list.toTuple();
    defer tuple.obj.decref();

    try std.testing.expectEqual(@as(usize, 4), tuple.length());
}

test "PyList setOwnedItem" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    var list = try PyList(root).new(2);
    defer list.obj.decref();
    const py1 = try py.create(root, 1);
    defer py1.decref();
    try list.setOwnedItem(0, py1);
    const py2 = try py.create(root, 2);
    defer py2.decref();
    try list.setOwnedItem(1, py2);

    try std.testing.expectEqual(@as(u8, 1), try list.getItem(u8, 0));
    try std.testing.expectEqual(@as(u8, 2), try list.getItem(u8, 1));
}

test "PyList fromSlice and toSlice - basic types" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    // Test i64 slice
    const ints = [_]i64{ 1, 2, 3, 4, 5 };
    const list = try PyList(root).fromSlice(&ints);
    defer list.obj.decref();

    try testing.expectEqual(@as(usize, 5), list.length());
    try testing.expectEqual(@as(i64, 3), try list.getItem(i64, 2));

    // Convert back
    const result = try list.toSlice(i64);
    defer py.allocator.free(result);

    try testing.expectEqual(@as(usize, 5), result.len);
    try testing.expectEqual(@as(i64, 1), result[0]);
    try testing.expectEqual(@as(i64, 5), result[4]);

    // Test f64 slice
    const floats = [_]f64{ 1.1, 2.2, 3.3 };
    const flist = try PyList(root).fromSlice(&floats);
    defer flist.obj.decref();

    const fresult = try flist.toSlice(f64);
    defer py.allocator.free(fresult);

    try testing.expectEqual(@as(f64, 2.2), fresult[1]);
}

test "PyList fromSlice - empty slice" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    const empty = [_]i64{};
    const list = try PyList(root).fromSlice(&empty);
    defer list.obj.decref();

    try testing.expectEqual(@as(usize, 0), list.length());

    const result = try list.toSlice(i64);
    defer py.allocator.free(result);

    try testing.expectEqual(@as(usize, 0), result.len);
}

test "PyList create and as - high level API" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    // Test create
    const floats = [_]f64{ 1.1, 2.2, 3.3 };
    const list = try PyList(root).create(&floats);
    defer list.obj.decref();

    try testing.expectEqual(@as(usize, 3), list.length());

    // Test as
    const result = try list.as([]f64);
    defer py.allocator.free(result);

    try testing.expectEqual(@as(usize, 3), result.len);
    try testing.expectEqual(@as(f64, 2.2), result[1]);
}
