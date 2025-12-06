# Debugging Guide

Complete guide to debugging Ziggy pyZ3 extensions with comprehensive tooling support.

---

## Overview

pyZ3 provides extensive debugging support including:

- ✅ **Debug logging** with configurable levels
- ✅ **Mixed Python/Zig stack traces**
- ✅ **Breakpoint support** for interactive debugging
- ✅ **Performance timing** and profiling
- ✅ **Memory inspection** utilities
- ✅ **Reference count tracking**
- ✅ **LLDB/GDB integration**
- ✅ **VSCode debugging** configurations
- ✅ **Core dump support**

---

## Quick Start

### Enable Debug Logging

**From Zig:**
```zig
const py = @import("pyz3");

pub fn my_function() void {
    // Enable debug mode
    py.debug.enableDebug();
    py.debug.setLogLevel(.debug);

    py.debug.info("Function called", .{});
    py.debug.debug("Detailed debug info", .{});
}
```

**From Python:**
```python
import my_extension

# Enable debug logging
my_extension.enable_debugging("debug")

# Your code here
my_extension.my_function()

# Disable when done
my_extension.disable_debugging()
```

---

## Debug Logging

### Log Levels

pyZ3 supports 5 log levels (from most to least verbose):

| Level | Usage | Example |
|-------|-------|---------|
| `trace` | Very detailed tracing | Loop iterations, state changes |
| `debug` | Debug information | Function entry/exit, variable values |
| `info` | General information | Operation start/complete |
| `warn` | Warnings | Deprecated features, recoverable errors |
| `err` | Errors | Exceptions, failures |

### Usage

```zig
const py = @import("pyz3");

pub fn calculate(args: struct { x: i64, y: i64 }) !i64 {
    py.debug.info("calculate called with x={d}, y={d}", .{ args.x, args.y });

    if (args.y == 0) {
        py.debug.err("Division by zero attempted!", .{});
        return error.DivisionByZero;
    }

    const result = args.x / args.y;
    py.debug.debug("Result: {d}", .{result});

    return result;
}
```

**Output:**
```
[pyZ3 INFO] calculate called with x=10, y=2
[pyZ3 DEBUG] Result: 5
```

---

## Stack Traces

### Print Mixed Stack Trace

```zig
py.debug.printStackTrace();
```

**Output:**
```
=== Stack Trace ===

--- Zig Stack ---
#0: 0x0000000100001234
#1: 0x0000000100001456
...

--- Python Stack ---
  File "test.py", line 42, in main
    result = my_extension.calculate(10, 0)
  ...

===================
```

### Python Exception Traceback

```zig
pub fn might_fail() !py.PyString {
    if (some_error) {
        py.debug.printException();
        return error.Failed;
    }
    return py.PyString.create("Success");
}
```

---

## Performance Timing

### Basic Timing

```zig
pub fn expensive_operation() !void {
    const timer = py.debug.Timer.start("expensive_operation");
    defer timer.stop();

    // Your code here
    for (0..1000000) |i| {
        // Work...
    }
}
```

**Output:**
```
[pyZ3 INFO] expensive_operation took 234ms
```

### Multiple Timers

```zig
pub fn multi_step_process() !void {
    {
        const timer1 = py.debug.Timer.start("step 1");
        defer timer1.stop();
        // Step 1 work
    }

    {
        const timer2 = py.debug.Timer.start("step 2");
        defer timer2.stop();
        // Step 2 work
    }
}
```

---

## Memory Debugging

### Inspect Memory

```zig
pub fn debug_buffer() void {
    const data = [_]u8{ 0x48, 0x65, 0x6c, 0x6c, 0x6f };  // "Hello"
    py.debug.inspectMemory(&data, data.len);
}
```

**Output:**
```
Memory at 0x7fff5fbff800:
  00000000: 48 65 6c 6c 6f                   | Hello
```

### Reference Count Inspection

```zig
pub fn check_refcount(obj: py.PyObject) void {
    py.debug.inspectRefCount(obj);
}
```

**Output:**
```
PyObject refcount: 3
```

Or with warnings:
```
PyObject refcount: 0
⚠️  WARNING: Reference count is 0! Object may be freed.
```

---

## Debug Assertions

### Safe Assertions

```zig
pub fn divide(args: struct { a: i64, b: i64 }) !f64 {
    py.debug.assertDebug(
        args.b != 0,
        "Denominator cannot be zero! numerator={d}",
        .{args.a}
    );

    return @as(f64, @floatFromInt(args.a)) / @as(f64, @floatFromInt(args.b));
}
```

If assertion fails:
```
❌ Assertion failed: Denominator cannot be zero! numerator=42

=== Stack Trace ===
...
```

---

## Debug Context

Track state across complex operations:

```zig
pub fn complex_task(args: struct { mode: []const u8 }) !void {
    var ctx = py.debug.DebugContext.init(py.allocator, "complex_task");
    defer ctx.deinit();

    try ctx.set("mode", args.mode);
    try ctx.set("step", "initialization");
    ctx.dump();

    // Do work...

    try ctx.set("step", "processing");
    ctx.dump();

    // More work...

    try ctx.set("step", "complete");
    ctx.dump();
}
```

**Output:**
```
=== Debug Context: complex_task ===
  mode = fast
  step = initialization
===========================

=== Debug Context: complex_task ===
  mode = fast
  step = processing
===========================
```

---

## Interactive Debugging

### Breakpoints

```zig
pub fn problematic_function() void {
    // ... some code ...

    py.debug.breakpoint();  // Pauses here in debug builds

    // ... more code ...
}
```

### Python Breakpoints

```python
from pyz3.debug import breakpoint_here

def my_test():
    # ... code ...

    breakpoint_here("About to call Zig function")

    result = my_extension.problematic_function()
```

**Output:**
```
🔴 About to call Zig function
   PID: 12345
   Attach debugger or press Enter to continue...
```

---

## Debugger Integration

### LLDB Debugging

#### 1. Start Python with LLDB

```bash
lldb -- python my_script.py
```

#### 2. Set Breakpoints

```lldb
(lldb) breakpoint set --name my_zig_function
(lldb) breakpoint set --file my_module.zig --line 42
(lldb) run
```

#### 3. Inspect Variables

```lldb
(lldb) frame variable
(lldb) print my_variable
(lldb) bt  # Backtrace
```

### GDB Debugging

#### 1. Start Python with GDB

```bash
gdb --args python my_script.py
```

#### 2. Set Breakpoints

```gdb
(gdb) break my_zig_function
(gdb) break my_module.zig:42
(gdb) run
```

#### 3. Inspect Variables

```gdb
(gdb) info locals
(gdb) print my_variable
(gdb) bt  # Backtrace
```

### Attach to Running Process

**LLDB:**
```bash
# Get PID from Python
python -c "import os; print(os.getpid()); input('Press Enter...')"

# In another terminal
lldb -p <PID>
```

**GDB:**
```bash
gdb -p <PID>
```

---

## VSCode Debugging

### Setup

1. **Install Extensions:**
   - Python
   - CodeLLDB (for native debugging)

2. **Configuration** (already included in `.vscode/launch.json`):

#### Debug Python Code
```json
{
    "name": "Python: Debug Extension",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "env": {
        "pyz3_DEBUG": "1"
    }
}
```

#### Debug Tests
```json
{
    "name": "Python: Debug Tests",
    "type": "python",
    "request": "launch",
    "module": "pytest",
    "args": ["-v", "-s", "${file}"]
}
```

#### Debug Native Code
```json
{
    "name": "LLDB: Attach to Python",
    "type": "lldb",
    "request": "attach",
    "pid": "${command:pickProcess}"
}
```

### Usage

1. Set breakpoints in Python or Zig code
2. Press F5 or select "Debug: Start Debugging"
3. Select configuration from dropdown
4. Code will pause at breakpoints

---

## Python Debugging Helpers

### Inspect Extension

```python
from pyz3.debug import inspect_extension

# Get detailed info about an extension
inspect_extension('my_extension')
```

**Output:**
```
======================================================================
Extension Module: my_extension
======================================================================

📁 Path: /path/to/my_extension.abi3.so
   Size: 234,567 bytes

🔍 Debug Symbols:
   ✅ Debug symbols present

📄 File Info:
   Mach-O 64-bit dynamically linked shared library x86_64

🐛 Debugger Commands:
   LLDB:
   $ lldb -p 12345
   (lldb) image add /path/to/my_extension.abi3.so

   GDB:
   $ gdb -p 12345
   (gdb) add-symbol-file /path/to/my_extension.abi3.so

======================================================================
```

### Enable Core Dumps

```python
from pyz3.debug import DebugHelper

# Enable core dumps for crash analysis
DebugHelper.enable_core_dumps()
```

### Create Debug Session

```python
from pyz3.debug import create_debug_session_script

# Generate a debug script
script = create_debug_session_script('my_extension', 'debug_session.py')

# Run it
# $ python debug_session.py
```

---

## Debugging Common Issues

### Memory Leaks

```zig
test "detect memory leaks" {
    var fixture = py.testing.TestFixture.init();
    defer fixture.deinit();  // Auto-detects leaks!

    py.debug.enableDebug();

    // Your test code
}
```

See [Memory Leak Detection](new_features.md#memory-leak-detection) for more.

### Reference Count Issues

```zig
pub fn track_refcounts(obj: py.PyObject) void {
    py.debug.inspectRefCount(obj);  // Initial: 1

    obj.incref();
    py.debug.inspectRefCount(obj);  // Now: 2

    obj.decref();
    py.debug.inspectRefCount(obj);  // Back to: 1
}
```

### Crashes

1. **Enable Core Dumps:**
   ```python
   from pyz3.debug import DebugHelper
   DebugHelper.enable_core_dumps()
   ```

2. **Run Program:**
   ```bash
   python my_script.py  # Crashes and creates core file
   ```

3. **Analyze Core:**
   ```bash
   lldb python -c core
   (lldb) bt  # See crash backtrace
   ```

### Slow Performance

```zig
pub fn slow_function() void {
    const timer = py.debug.Timer.start("slow_function");
    defer timer.stop();

    // Identify slow sections
    {
        const t1 = py.debug.Timer.start("section 1");
        defer t1.stop();
        // Code...
    }

    {
        const t2 = py.debug.Timer.start("section 2");
        defer t2.stop();
        // Code...
    }
}
```

---

## Best Practices

### 1. Use Appropriate Log Levels

```zig
// ✅ Good
py.debug.trace("Loop iteration {d}", .{i});  // Very detailed
py.debug.info("Operation complete", .{});    // General info

// ❌ Bad
py.debug.info("Loop iteration {d}", .{i});   // Too verbose
```

### 2. Disable Debug in Production

```zig
pub fn init() void {
    if (builtin.mode == .Debug) {
        py.debug.enableDebug();
    }
}
```

### 3. Use Timers for Performance

```zig
// Always wrap expensive operations
const timer = py.debug.Timer.start("operation_name");
defer timer.stop();
```

### 4. Check Reference Counts

```zig
// When debugging lifetime issues
py.debug.inspectRefCount(obj);
```

### 5. Use Debug Context for State

```zig
// Track state in complex operations
var ctx = py.debug.DebugContext.init(py.allocator, "operation");
defer ctx.deinit();
```

---

## Configuration Files

### .lldbinit

Place in project root or `~/.lldbinit`:

```lldb
# pyZ3-specific helpers
command regex pyz3-break 's/(.+)/breakpoint set --name %1/'

# Use it:
# (lldb) pyz3-break my_function
```

### .gdbinit

Place in project root or `~/.gdbinit`:

```gdb
# pyZ3-specific helpers
define pyz3-break
    break $arg0
end

# Use it:
# (gdb) pyz3-break my_function
```

### .vscode/launch.json

Pre-configured debugging profiles (see [VSCode Debugging](#vscode-debugging) section).

---

## Examples

Complete examples available in `example/debugging.zig`:

```bash
# Run examples
poetry run pytest example/debugging.zig -v
```

---

## Troubleshooting

### Debug Symbols Missing

**Problem**: Debugger can't find symbols.

**Solution**:
```bash
# Build with debug symbols
zig build -Doptimize=Debug

# Verify symbols exist
file my_extension.so
# Should show: "not stripped" or "with debug_info"
```

### Breakpoints Not Hit

**Problem**: Breakpoints don't trigger.

**Solution**:
1. Ensure debug build: `zig build -Doptimize=Debug`
2. Load symbols in debugger:
   ```lldb
   (lldb) image add /path/to/extension.so
   ```

### No Debug Output

**Problem**: Debug logging not appearing.

**Solution**:
```zig
// Make sure debug is enabled
py.debug.enableDebug();
py.debug.setLogLevel(.trace);  // Set verbose level
```

---

## Summary

Ziggy pyZ3 provides comprehensive debugging support:

✅ Multi-level debug logging
✅ Performance timing
✅ Memory inspection
✅ Reference count tracking
✅ Stack traces (Python + Zig)
✅ LLDB/GDB integration
✅ VSCode debugging
✅ Core dump support
✅ Interactive breakpoints

See also:
- [Memory Leak Detection](new_features.md#memory-leak-detection)
- [Testing Guide](_4_testing.md)
- [Examples](../../example/debugging.zig)
