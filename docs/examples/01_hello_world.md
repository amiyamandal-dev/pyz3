# Example 1: Hello World

The simplest possible PyZ3 extension.

## Files

### src/hello_world.zig
```zig
const py = @import("pyz3");

/// Simple greeting function
pub fn hello(name: py.PyString) !py.PyString {
    const name_str = try name.asSlice();
    const greeting = try std.fmt.allocPrint(
        py.allocator,
        "Hello, {s}!",
        .{name_str}
    );
    defer py.allocator.free(greeting);
    return py.PyString.fromSlice(greeting);
}

comptime {
    py.rootmodule(@This());
}
```

### pyproject.toml
```toml
[build-system]
requires = ["pyz3>=0.8.0"]
build-backend = "pyz3.build"

[project]
name = "hello-world-ext"
version = "0.1.0"

[tool.pyz3.ext-module.hello_world]
root = "src/hello_world.zig"
```

### test_hello.py
```python
from hello_world import hello

def test_hello():
    result = hello("World")
    assert result == "Hello, World!"

def test_hello_unicode():
    result = hello("世界")
    assert "世界" in result
```

## Build and Run

```bash
# Build
python -m pyz3 build

# Test
python -c "from hello_world import hello; print(hello('PyZ3'))"

# Run tests
pytest test_hello.py
```
