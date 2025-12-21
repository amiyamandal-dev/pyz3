const py = @import("pyz3");
const std = @import("std");

const root = @This();

// --8<-- [start:debug-logging]
/// Example function with debug logging
pub fn calculate_fibonacci(args: struct { n: u64 }) !u64 {
    py.debug.info("calculate_fibonacci called with n={d}", .{args.n});

    if (args.n < 2) {
        py.debug.trace("Base case: n={d}, returning {d}", .{ args.n, args.n });
        return args.n;
    }

    var sum: u64 = 0;
    var last: u64 = 0;
    var curr: u64 = 1;

    py.debug.debug("Starting fibonacci loop for n={d}", .{args.n});

    for (1..args.n) |i| {
        sum = last + curr;
        last = curr;
        curr = sum;

        if (i % 10 == 0) {
            py.debug.trace("Progress: i={d}, current={d}", .{ i, curr });
        }
    }

    py.debug.info("calculate_fibonacci returning {d}", .{sum});
    return sum;
}
// --8<-- [end:debug-logging]

// --8<-- [start:timer-debug]
/// Example function with performance timing
pub fn slow_operation(args: struct { iterations: u32 }) !py.PyString {
    const timer = py.debug.Timer.start("slow_operation");
    defer timer.stop();

    py.debug.info("Starting slow operation with {d} iterations", .{args.iterations});

    var result: u64 = 0;
    for (0..args.iterations) |i| {
        result +%= i;
        // Simulate work
        if (i % 1000 == 0) {
            std.time.sleep(1000); // 1 microsecond
        }
    }

    py.debug.info("Operation completed, result={d}", .{result});
    return py.PyString.createFmt("Result: {d}", .{result});
}
// --8<-- [end:timer-debug]

// --8<-- [start:refcount-debug]
/// Example showing reference count debugging
pub fn test_refcounts(args: struct { value: py.PyObject }) !void {
    py.debug.info("Initial object state:", .{});
    py.debug.inspectRefCount(args.value);

    // Increment reference
    args.value.incref();
    py.debug.info("After incref:", .{});
    py.debug.inspectRefCount(args.value);

    // Decrement reference
    args.value.decref();
    py.debug.info("After decref:", .{});
    py.debug.inspectRefCount(args.value);
}
// --8<-- [end:refcount-debug]

// --8<-- [start:exception-debug]
/// Example showing exception debugging
pub fn might_fail(args: struct { should_fail: bool }) !py.PyString {
    py.debug.info("might_fail called with should_fail={}", .{args.should_fail});

    if (args.should_fail) {
        py.debug.warn("About to raise exception", .{});
        py.debug.printStackTrace();
        return py.ValueError(root).raise("This is a test exception");
    }

    return py.PyString.create("Success!");
}
// --8<-- [end:exception-debug]

// --8<-- [start:memory-inspect]
/// Example showing memory inspection
pub fn inspect_data() !void {
    const data = [_]u8{ 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x57, 0x6f, 0x72, 0x6c, 0x64, 0x00 };

    py.debug.info("Inspecting memory:", .{});
    py.debug.inspectMemory(&data, data.len);
}
// --8<-- [end:memory-inspect]

// --8<-- [start:debug-context]
/// Example using debug context for state tracking
pub fn complex_operation(args: struct { mode: []const u8 }) !py.PyString {
    var ctx = py.debug.DebugContext.init(py.allocator, "complex_operation");
    defer ctx.deinit();

    try ctx.set("mode", args.mode);
    try ctx.set("start_time", "now");

    py.debug.info("Starting complex operation", .{});
    ctx.dump();

    // Do complex work
    var result: []const u8 = "unknown";
    if (std.mem.eql(u8, args.mode, "fast")) {
        result = "fast_result";
    } else if (std.mem.eql(u8, args.mode, "slow")) {
        result = "slow_result";
    }

    try ctx.set("result", result);
    ctx.dump();

    return py.PyString.createFmt("Result: {s}", .{result});
}
// --8<-- [end:debug-context]

// --8<-- [start:assert-debug]
/// Example using debug assertions
pub fn validated_divide(args: struct { numerator: i64, denominator: i64 }) !f64 {
    py.debug.assertDebug(args.denominator != 0, "Denominator cannot be zero! numerator={d}", .{args.numerator});

    const result: f64 = @as(f64, @floatFromInt(args.numerator)) / @as(f64, @floatFromInt(args.denominator));

    py.debug.debug("Division result: {d} / {d} = {d}", .{ args.numerator, args.denominator, result });

    return result;
}
// --8<-- [end:assert-debug]

/// Enable debug mode from Python
pub fn enable_debugging(args: struct { level: ?[]const u8 }) void {
    py.debug.enableDebug();

    if (args.level) |lvl| {
        if (std.mem.eql(u8, lvl, "trace")) {
            py.debug.setLogLevel(.trace);
        } else if (std.mem.eql(u8, lvl, "debug")) {
            py.debug.setLogLevel(.debug);
        } else if (std.mem.eql(u8, lvl, "info")) {
            py.debug.setLogLevel(.info);
        } else if (std.mem.eql(u8, lvl, "warn")) {
            py.debug.setLogLevel(.warn);
        } else if (std.mem.eql(u8, lvl, "error")) {
            py.debug.setLogLevel(.err);
        }
    }

    py.debug.info("Debug mode enabled at level: {s}", .{py.debug.getLogLevel().toString()});
}

/// Disable debug mode from Python
pub fn disable_debugging() void {
    py.debug.info("Disabling debug mode", .{});
    py.debug.disableDebug();
}

comptime {
    py.rootmodule(root);
}

// --8<-- [start:test-debug-logging]
test "debug logging levels" {
    var fixture = py.testing.TestFixture.init();
    defer fixture.deinit();

    fixture.initPython();

    // Enable debug mode
    py.debug.enableDebug();
    defer py.debug.disableDebug();

    // Test different log levels
    py.debug.setLogLevel(.trace);
    py.debug.trace("This is a trace message", .{});
    py.debug.debug("This is a debug message", .{});
    py.debug.info("This is an info message", .{});
    py.debug.warn("This is a warning message", .{});
    py.debug.err("This is an error message", .{});

    // Test level filtering
    py.debug.setLogLevel(.warn);
    py.debug.debug("This should NOT appear", .{});
    py.debug.warn("This SHOULD appear", .{});
}
// --8<-- [end:test-debug-logging]

// --8<-- [start:test-timer]
test "performance timing" {
    var fixture = py.testing.TestFixture.init();
    defer fixture.deinit();

    py.debug.enableDebug();
    defer py.debug.disableDebug();

    const timer = py.debug.Timer.start("test operation");
    defer timer.stop();

    // Do some work
    var sum: u64 = 0;
    for (0..1000) |i| {
        sum +%= i;
    }

    try std.testing.expect(sum > 0);
}
// --8<-- [end:test-timer]

// --8<-- [start:test-memory-inspection]
test "memory inspection" {
    var fixture = py.testing.TestFixture.init();
    defer fixture.deinit();

    py.debug.enableDebug();
    defer py.debug.disableDebug();

    const data = [_]u8{ 0xDE, 0xAD, 0xBE, 0xEF };
    py.debug.inspectMemory(&data, data.len);
}
// --8<-- [end:test-memory-inspection]
