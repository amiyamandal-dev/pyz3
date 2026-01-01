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
const PySequenceMixin = @import("./sequence.zig").PySequenceMixin;
const ffi = py.ffi;
const PyObject = py.PyObject;
const PyLong = py.PyLong;
const PyError = @import("../errors.zig").PyError;
const State = @import("../discovery.zig").State;

/// Wrapper for Python PyList.
/// See: https://docs.python.org/3/c-api/list.html
///
/// This type provides full Python list compatibility with all standard methods.
/// PySequenceMixin provides additional sequence protocol operations like contains(),
/// index(), count(), concat(), etc., but cannot be directly integrated via usingnamespace
/// due to Zig 0.15 limitations with extern structs.
pub fn PyList(comptime root: type) type {
    return extern struct {
        obj: PyObject,

        const Self = @This();
        pub const from = PyObjectMixin("list", "PyList", Self);

        // ============================================================
        // Construction
        // ============================================================

        /// Create a new list with the specified size.
        /// All items will be initialized to None.
        pub fn new(size: usize) !Self {
            // Check for integer overflow before casting to isize
            if (size > std.math.maxInt(isize)) {
                return PyError.PyRaised; // Python will raise OverflowError
            }
            const list = ffi.PyList_New(@intCast(size)) orelse return PyError.PyRaised;
            return .{ .obj = .{ .py = list } };
        }

        /// Create a new list from an iterable.
        /// Equivalent to Python: list(iterable)
        pub fn fromIterable(iterable: anytype) !Self {
            const iterableObj = py.object(root, iterable);
            const list = ffi.PySequence_List(iterableObj.py) orelse return PyError.PyRaised;
            return .{ .obj = .{ .py = list } };
        }

        // ============================================================
        // Basic Operations
        // ============================================================

        /// Get the length of the list.
        /// Equivalent to Python: len(list)
        pub fn length(self: Self) usize {
            const size = ffi.PyList_Size(self.obj.py);
            // PyList_Size returns isize, which should always be >= 0 for valid lists
            if (size < 0) return 0; // Error case, but we can't return error from this function
            return @intCast(size);
        }

        /// Get item at index (returns borrowed reference).
        /// Supports negative indices (Python-style: -1 is last element).
        /// Equivalent to Python: list[index]
        pub fn getItem(self: Self, comptime T: type, idx: isize) !T {
            if (ffi.PyList_GetItem(self.obj.py, idx)) |item| {
                return py.as(root, T, py.PyObject{ .py = item });
            } else {
                return PyError.PyRaised;
            }
        }

        /// Get a slice of the list (returns new reference).
        /// Equivalent to Python: list[start:end]
        pub fn getSlice(self: Self, low: isize, high: isize) !Self {
            if (ffi.PyList_GetSlice(self.obj.py, low, high)) |item| {
                return .{ .obj = .{ .py = item } };
            } else {
                return PyError.PyRaised;
            }
        }

        /// Set slice of the list.
        /// Equivalent to Python: list[start:end] = values
        pub fn setSlice(self: Self, start: isize, end: isize, values: anytype) !void {
            const py_values = py.object(root, values);
            if (ffi.PySequence_SetSlice(self.obj.py, start, end, py_values.py) < 0) {
                return PyError.PyRaised;
            }
        }

        /// This function "steals" a reference to item and discards a reference to an item already in the list at the affected position.
        /// Use this when you have an owned reference that you want to transfer to the list.
        pub fn setOwnedItem(self: Self, pos: usize, value: anytype) !void {
            // Since this function steals the reference, it can only accept object-like values.
            if (ffi.PyList_SetItem(self.obj.py, @intCast(pos), py.object(root, value).py) < 0) {
                return PyError.PyRaised;
            }
        }

        /// Set the item at the given position.
        /// Equivalent to Python: list[index] = value
        pub fn setItem(self: Self, pos: usize, value: anytype) !void {
            const valueObj = try py.create(root, value);
            // setOwnedItem steals a reference, but if it fails we need to clean up
            errdefer valueObj.decref();
            return self.setOwnedItem(pos, valueObj);
        }

        /// Delete item at index.
        /// Equivalent to Python: del list[index]
        pub fn delItem(self: Self, idx: isize) !void {
            if (ffi.PySequence_DelItem(self.obj.py, idx) < 0) {
                return PyError.PyRaised;
            }
        }

        /// Delete slice of the list.
        /// Equivalent to Python: del list[start:end]
        pub fn delSlice(self: Self, start: isize, end: isize) !void {
            if (ffi.PySequence_DelSlice(self.obj.py, start, end) < 0) {
                return PyError.PyRaised;
            }
        }

        // ============================================================
        // Modification Operations
        // ============================================================

        /// Insert the item into list in front of index idx.
        /// Equivalent to Python: list.insert(index, value)
        /// Note: PyList_Insert increments the reference count, so we don't decref
        pub fn insert(self: Self, idx: isize, value: anytype) !void {
            const valueObj = try py.create(root, value);
            // PyList_Insert increments refcount, so we don't need to keep our reference
            // But we should clean up if insertion fails
            if (ffi.PyList_Insert(self.obj.py, idx, valueObj.py) < 0) {
                valueObj.decref();
                return PyError.PyRaised;
            }
            // Success: list now owns the reference, we can release ours
            valueObj.decref();
        }

        /// Append the object at the end of list.
        /// Equivalent to Python: list.append(value)
        /// Note: PyList_Append increments the reference count, so we don't decref
        pub fn append(self: Self, value: anytype) !void {
            const valueObj = try py.create(root, value);
            // PyList_Append increments refcount, so we don't need to keep our reference
            // But we should clean up if append fails
            if (ffi.PyList_Append(self.obj.py, valueObj.py) < 0) {
                valueObj.decref();
                return PyError.PyRaised;
            }
            // Success: list now owns the reference, we can release ours
            valueObj.decref();
        }

        /// Extend the list by appending all items from iterable.
        /// Equivalent to Python: list.extend(iterable)
        pub fn extend(self: Self, iterable: anytype) !void {
            const iterableObj = py.object(root, iterable);
            // Use PySequence_InPlaceConcat for efficient extension
            const result = ffi.PySequence_InPlaceConcat(self.obj.py, iterableObj.py) orelse return PyError.PyRaised;
            // In-place concat may return a new object, but for lists it typically returns the same object
            // We don't need to update self.obj.py as it should be the same reference
            _ = result;
        }

        /// Remove and return item at index (default last).
        /// Equivalent to Python: list.pop([index])
        pub fn pop(self: Self, comptime T: type, idx_opt: ?isize) !T {
            const len_val = self.length();
            if (len_val == 0) {
                return py.IndexError(root).raise("pop from empty list");
            }

            const idx = if (idx_opt) |i| blk: {
                // Normalize negative index
                const normalized: isize = if (i < 0) @as(isize, @intCast(len_val)) + i else i;
                if (normalized < 0 or normalized >= @as(isize, @intCast(len_val))) {
                    return py.IndexError(root).raise("pop index out of range");
                }
                break :blk normalized;
            } else @as(isize, @intCast(len_val)) - 1;

            // Get the item before deleting (returns new reference)
            const item = ffi.PySequence_GetItem(self.obj.py, idx) orelse return PyError.PyRaised;
            defer ffi.Py_DecRef(item);

            // Delete the item
            if (ffi.PySequence_DelItem(self.obj.py, idx) < 0) {
                return PyError.PyRaised;
            }

            // Convert and return (item is already decref'd above, but py.as may create new ref)
            return py.as(root, T, py.PyObject{ .py = item });
        }

        /// Remove first occurrence of value.
        /// Equivalent to Python: list.remove(value)
        /// Raises ValueError if value is not found.
        pub fn remove(self: Self, value: anytype) !void {
            const valueObj = try py.create(root, value);
            defer valueObj.decref();

            // Find the index
            const idx = ffi.PySequence_Index(self.obj.py, valueObj.py);
            if (idx < 0) {
                return PyError.PyRaised; // ValueError raised by Python
            }

            // Delete the item at that index
            if (ffi.PySequence_DelItem(self.obj.py, idx) < 0) {
                return PyError.PyRaised;
            }
        }

        /// Remove all items from the list.
        /// Equivalent to Python: list.clear()
        pub fn clear(self: Self) !void {
            // Use PySequence_DelSlice to delete all items
            const len_val: isize = @intCast(self.length());
            if (len_val > 0) {
                if (ffi.PySequence_DelSlice(self.obj.py, 0, len_val) < 0) {
                    return PyError.PyRaised;
                }
            }
        }

        /// Create a shallow copy of the list.
        /// Equivalent to Python: list.copy() or list[:]
        pub fn copy(self: Self) !Self {
            // Use getSlice to get a copy of the entire list
            return self.getSlice(0, @as(isize, @intCast(self.length())));
        }

        // ============================================================
        // Search Operations
        // ============================================================

        /// Find index of first occurrence of value.
        /// Equivalent to Python: list.index(value)
        /// Raises ValueError if value is not found.
        pub fn index(self: Self, value: anytype) !usize {
            const valueObj = try py.create(root, value);
            defer valueObj.decref();

            const idx = ffi.PySequence_Index(self.obj.py, valueObj.py);
            if (idx < 0) {
                return PyError.PyRaised; // ValueError raised by Python
            }
            return @intCast(idx);
        }

        /// Count occurrences of value in list.
        /// Equivalent to Python: list.count(value)
        pub fn count(self: Self, value: anytype) !usize {
            const valueObj = try py.create(root, value);
            defer valueObj.decref();

            const cnt = ffi.PySequence_Count(self.obj.py, valueObj.py);
            if (cnt < 0) {
                return PyError.PyRaised;
            }
            return @intCast(cnt);
        }

        /// Check if list contains value.
        /// Equivalent to Python: value in list
        pub fn contains(self: Self, value: anytype) !bool {
            const valueObj = try py.create(root, value);
            defer valueObj.decref();

            const result = ffi.PySequence_Contains(self.obj.py, valueObj.py);
            if (result < 0) {
                return PyError.PyRaised;
            }
            return result == 1;
        }

        // ============================================================
        // Sorting and Reversal
        // ============================================================

        /// Sort the items of list in place.
        /// Equivalent to Python: list.sort()
        pub fn sort(self: Self) !void {
            if (ffi.PyList_Sort(self.obj.py) < 0) {
                return PyError.PyRaised;
            }
        }

        /// Reverse the items of list in place.
        /// Equivalent to Python: list.reverse()
        pub fn reverse(self: Self) !void {
            if (ffi.PyList_Reverse(self.obj.py) < 0) {
                return PyError.PyRaised;
            }
        }

        // ============================================================
        // Conversion Operations
        // ============================================================

        /// Convert list to tuple.
        /// Equivalent to Python: tuple(list)
        pub fn toTuple(self: Self) !py.PyTuple(root) {
            const pytuple = ffi.PyList_AsTuple(self.obj.py) orelse return PyError.PyRaised;
            return py.PyTuple(root).from.unchecked(.{ .py = pytuple });
        }
    };
}

const testing = std.testing;

test "PyList basic operations" {
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

test "PyList extend" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    var list1 = try PyList(root).new(0);
    defer list1.obj.decref();
    try list1.append(1);
    try list1.append(2);

    var list2 = try PyList(root).new(0);
    defer list2.obj.decref();
    try list2.append(3);
    try list2.append(4);

    try list1.extend(list2);
    try testing.expectEqual(@as(usize, 4), list1.length());
    try testing.expectEqual(@as(i64, 3), try list1.getItem(i64, 2));
    try testing.expectEqual(@as(i64, 4), try list1.getItem(i64, 3));
}

test "PyList pop" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    var list = try PyList(root).new(0);
    defer list.obj.decref();
    try list.append(1);
    try list.append(2);
    try list.append(3);

    const popped = try list.pop(i64, null);
    try testing.expectEqual(@as(i64, 3), popped);
    try testing.expectEqual(@as(usize, 2), list.length());

    const popped2 = try list.pop(i64, 0);
    try testing.expectEqual(@as(i64, 1), popped2);
    try testing.expectEqual(@as(usize, 1), list.length());
}

test "PyList remove" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    var list = try PyList(root).new(0);
    defer list.obj.decref();
    try list.append(1);
    try list.append(2);
    try list.append(3);
    try list.append(2);

    try list.remove(2);
    try testing.expectEqual(@as(usize, 3), list.length());
    try testing.expectEqual(@as(i64, 1), try list.getItem(i64, 0));
    try testing.expectEqual(@as(i64, 3), try list.getItem(i64, 1));
    try testing.expectEqual(@as(i64, 2), try list.getItem(i64, 2));
}

test "PyList clear" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    var list = try PyList(root).new(0);
    defer list.obj.decref();
    try list.append(1);
    try list.append(2);
    try list.append(3);

    try list.clear();
    try testing.expectEqual(@as(usize, 0), list.length());
}

test "PyList copy" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    var list1 = try PyList(root).new(0);
    defer list1.obj.decref();
    try list1.append(1);
    try list1.append(2);
    try list1.append(3);

    var list2 = try list1.copy();
    defer list2.obj.decref();

    try testing.expectEqual(@as(usize, 3), list2.length());
    try list1.append(4);
    try testing.expectEqual(@as(usize, 4), list1.length());
    try testing.expectEqual(@as(usize, 3), list2.length());
}

test "PyList index and count" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    var list = try PyList(root).new(0);
    defer list.obj.decref();
    try list.append(1);
    try list.append(2);
    try list.append(3);
    try list.append(2);

    const idx = try list.index(2);
    try testing.expectEqual(@as(usize, 1), idx);

    const cnt = try list.count(2);
    try testing.expectEqual(@as(usize, 2), cnt);

    const cnt2 = try list.count(99);
    try testing.expectEqual(@as(usize, 0), cnt2);
}

test "PyList contains" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    var list = try PyList(root).new(0);
    defer list.obj.decref();
    try list.append(1);
    try list.append(2);
    try list.append(3);

    try testing.expect(try list.contains(2));
    try testing.expect(!try list.contains(99));
}

test "PyList slice operations" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    var list = try PyList(root).new(0);
    defer list.obj.decref();
    try list.append(1);
    try list.append(2);
    try list.append(3);
    try list.append(4);
    try list.append(5);

    var slice = try list.getSlice(1, 4);
    defer slice.obj.decref();
    try testing.expectEqual(@as(usize, 3), slice.length());
    try testing.expectEqual(@as(i64, 2), try slice.getItem(i64, 0));
    try testing.expectEqual(@as(i64, 4), try slice.getItem(i64, 2));

    var list2 = try PyList(root).new(0);
    defer list2.obj.decref();
    try list2.append(10);
    try list2.append(20);

    try list.setSlice(1, 3, list2);
    try testing.expectEqual(@as(i64, 10), try list.getItem(i64, 1));
    try testing.expectEqual(@as(i64, 20), try list.getItem(i64, 2));
}

test "PyList delItem and delSlice" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    var list = try PyList(root).new(0);
    defer list.obj.decref();
    try list.append(1);
    try list.append(2);
    try list.append(3);
    try list.append(4);
    try list.append(5);

    try list.delItem(1);
    try testing.expectEqual(@as(usize, 4), list.length());
    try testing.expectEqual(@as(i64, 3), try list.getItem(i64, 1));

    try list.delSlice(0, 2);
    try testing.expectEqual(@as(usize, 2), list.length());
    try testing.expectEqual(@as(i64, 4), try list.getItem(i64, 0));
}

test "PyList fromIterable" {
    py.initialize();
    defer py.finalize();

    const root = @This();

    var list1 = try PyList(root).new(0);
    defer list1.obj.decref();
    try list1.append(1);
    try list1.append(2);
    try list1.append(3);

    var list2 = try PyList(root).fromIterable(list1);
    defer list2.obj.decref();

    try testing.expectEqual(@as(usize, 3), list2.length());
    try testing.expectEqual(@as(i64, 1), try list2.getItem(i64, 0));
    try testing.expectEqual(@as(i64, 2), try list2.getItem(i64, 1));
    try testing.expectEqual(@as(i64, 3), try list2.getItem(i64, 2));
}
