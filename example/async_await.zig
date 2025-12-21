const py = @import("pyz3");
const std = @import("std");

const root = @This();

/// Call a Python coroutine from Zig
/// This function accepts a coroutine and sends values to it
pub fn call_coroutine(args: struct { coro: py.PyObject }) !py.PyObject {
    if (!py.PyCoroutine.check(args.coro)) {
        return py.TypeError(root).raise("Expected a coroutine");
    }

    const coro = py.PyCoroutine{ .obj = args.coro };

    // Send None to start the coroutine
    const result = try coro.send(null);

    return result;
}

/// Run an awaitable and return its result
/// This is a blocking operation that runs the awaitable to completion
pub fn run_awaitable(args: struct { awaitable: py.PyObject }) !py.PyObject {
    const awaitable = py.PyAwaitable{ .obj = args.awaitable };
    return try awaitable.await_();
}

/// Create a simple future-like object
/// This demonstrates how to interact with Python's async ecosystem
pub const SimpleFuture = py.class(struct {
    const Self = @This();

    result: ?py.PyObject = null,
    done: bool = false,

    pub fn __init__(self: *Self) void {
        self.* = .{};
    }

    pub fn __del__(self: *Self) void {
        if (self.result) |res| {
            res.decref();
        }
    }

    /// Set the result of the future
    pub fn set_result(self: *Self, args: struct { value: py.PyObject }) void {
        args.value.incref();
        self.result = args.value;
        self.done = true;
    }

    /// Check if the future is done
    pub fn is_done(self: *const Self) bool {
        return self.done;
    }

    /// Get the result (blocking if not done)
    pub fn get_result(self: *const Self) !py.PyObject {
        if (!self.done) {
            return py.RuntimeError(root).raise("Future not done yet");
        }

        if (self.result) |res| {
            res.incref();
            return res;
        }

        const none = py.PyObject{ .py = py.ffi.Py_None() };
        none.incref();
        return none;
    }

    /// Make this future awaitable by implementing __await__
    pub fn __await__(self: *const Self) !py.PyIter {
        // Return an iterator that yields until the future is done
        // For simplicity, this is a synchronous implementation
        if (self.done) {
            // Return a single-shot iterator that yields the result
            if (self.result) |res| {
                res.incref();
                // Create a single-item iterator
                const tuple = try py.tuple(root, .{res});
                return try tuple.iter();
            }
        }

        return py.RuntimeError(root).raise("Awaitable future not yet implemented fully");
    }
});

comptime {
    py.rootmodule(root);
}

// --8<-- [start:test-coroutine]
test "interact with Python coroutines" {
    var fixture = py.testing.TestFixture.init();
    defer fixture.deinit();

    fixture.initPython();

    // Import asyncio
    const asyncio = try py.import("asyncio");
    defer asyncio.decref();

    // Create a simple coroutine using Python
    const code =
        \\async def simple_coro():
        \\    return 42
        \\coro = simple_coro()
    ;

    const builtins = try py.import("builtins");
    defer builtins.decref();

    const exec = try builtins.getAttribute("exec");
    defer exec.decref();

    const globals = try py.dict(root);
    defer globals.decref();

    _ = try py.call(root, exec, .{ code, globals });

    const coro = try globals.getItem(try py.PyString.create("coro"));
    defer coro.decref();

    // Verify it's a coroutine
    try std.testing.expect(py.PyCoroutine.check(coro));
}
// --8<-- [end:test-coroutine]

// --8<-- [start:test-future]
test "SimpleFuture usage" {
    var fixture = py.testing.TestFixture.init();
    defer fixture.deinit();

    fixture.initPython();

    // Create a future
    const future_cls = try py.create(root, SimpleFuture);
    defer future_cls.decref();

    const future_inst = try py.call0(root, future_cls);
    defer future_inst.decref();

    // Initially not done
    const is_done_method = try future_inst.getAttribute("is_done");
    defer is_done_method.decref();

    const done1 = try py.call0(root, is_done_method);
    defer done1.decref();

    const done1_bool = try py.as(root, bool, done1);
    try std.testing.expect(!done1_bool);

    // Set result
    const set_result = try future_inst.getAttribute("set_result");
    defer set_result.decref();

    const value = try py.create(root, @as(i64, 123));
    defer value.decref();

    _ = try py.call(root, set_result, .{value});

    // Now it's done
    const done2 = try py.call0(root, is_done_method);
    defer done2.decref();

    const done2_bool = try py.as(root, bool, done2);
    try std.testing.expect(done2_bool);
}
// --8<-- [end:test-future]
