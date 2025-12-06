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

const py = @import("pyz3");
const std = @import("std");

const root = @This();

/// Example function that properly manages memory
pub fn safe_concat(args: struct { left: []const u8, right: []const u8 }) !py.PyString {
    return py.PyString.createFmt("{s}{s}", .{ args.left, args.right });
}

/// Example function with a potential memory leak (for demonstration)
pub fn leaky_function() !py.PyList(root) {
    const list = try py.PyList(root).new();

    // Allocate some memory
    const data = try py.allocator.alloc(u8, 100);
    // Intentionally forget to free - this would leak!
    // py.allocator.free(data);
    _ = data;

    return list;
}

comptime {
    py.rootmodule(root);
}

// --8<-- [start:test-with-fixture]
test "safe memory management" {
    var fixture = py.testing.TestFixture.init();
    defer fixture.deinit(); // Will check for leaks automatically

    fixture.initPython();

    // Use the test allocator
    const alloc = fixture.allocator();

    // Allocate and properly free memory
    const data = try alloc.alloc(u8, 100);
    defer alloc.free(data);

    // Test passes - no leaks detected
}
// --8<-- [end:test-with-fixture]

// --8<-- [start:test-string-operations]
test "string concatenation without leaks" {
    var fixture = py.testing.TestFixture.init();
    defer fixture.deinit();

    fixture.initPython();

    // This should not leak
    const result = try safe_concat(.{ .left = "Hello ", .right = "World" });
    defer result.obj.decref();

    const slice = try result.asSlice();
    try std.testing.expectEqualStrings("Hello World", slice);
}
// --8<-- [end:test-string-operations]

// --8<-- [start:test-leak-detection]
test "detecting memory leaks" {
    var fixture = py.testing.TestFixture.init();
    defer fixture.deinit(); // This will panic if there's a leak

    const alloc = fixture.allocator();

    // This will leak and be caught by deinit()
    _ = try alloc.alloc(u8, 50);
    // Forgot to call alloc.free() - leak!
}
// --8<-- [end:test-leak-detection]
