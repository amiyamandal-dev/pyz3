# Testing Guide for New Type Wrappers

This guide explains how to test the 13 new Python type wrappers implemented in pyZ3.

## Quick Start

### Run All Tests (Automated)

```bash
./test_new_types.sh --all
```

This will:
1. ✅ Check prerequisites (Zig, Python, pytest)
2. ✅ Build the project
3. ✅ Run Zig unit tests
4. ✅ Run Python type compatibility tests
5. ✅ Run integration tests
6. ✅ Run pytest suite
7. ✅ Show summary

### Interactive Menu (Recommended)

```bash
./test_new_types.sh
```

You'll see an interactive menu:
```
========================================
Ziggy pyZ3 - New Types Test Menu
========================================

1. Run all tests
2. Clean and rebuild
3. Run Zig tests only
4. Run Python type tests only
5. Run integration test
6. Run pytest
7. Show implementation summary
8. Quick verification
9. Exit

Choose an option (1-9):
```

## Available Commands

### Show Help
```bash
./test_new_types.sh --help
```

### Build Only
```bash
./test_new_types.sh --build
```

### Clean and Rebuild
```bash
./test_new_types.sh --clean
```

### Run Zig Tests Only
```bash
./test_new_types.sh --zig
```

### Run Python Tests Only
```bash
./test_new_types.sh --python
```

### Run Integration Test
```bash
./test_new_types.sh --integration
```

### Quick Verification (Fastest)
```bash
./test_new_types.sh --quick
```

Output:
```
Testing all new types...

✅ Complex: 5.0 == 5.0
✅ Decimal: 0.3 == 0.3
✅ DateTime: 2025 >= 2025
✅ Path: True == True
✅ UUID: 36 == 36
✅ Set: True == True
✅ Range: 10 == 10
✅ ByteArray: b'test' == b'test'

🎉 All types verified!
```

## Manual Testing

### 1. Build the Project

```bash
cd /Volumes/ssd/pyZ3
zig build
```

### 2. Run Python Interactive Tests

```bash
python3 << 'EOF'
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import uuid

# Test PyComplex
c = complex(3, 4)
print(f"Complex: {c}, abs={abs(c)}")

# Test PyDecimal
d = Decimal('0.1') + Decimal('0.2')
print(f"Decimal: 0.1 + 0.2 = {d}")

# Test PyDateTime
now = datetime.now()
print(f"DateTime: {now.isoformat()}")

# Test PyPath
p = Path.cwd()
print(f"Path: {p} exists={p.exists()}")

# Test PyUUID
u = uuid.uuid4()
print(f"UUID: {u}")

# Test PySet
s = {1, 2, 3} | {3, 4, 5}
print(f"Set union: {s}")

# Test PyRange
r = list(range(5))
print(f"Range: {r}")

# Test PyByteArray
ba = bytearray(b"Hello")
ba.extend(b" World")
print(f"ByteArray: {bytes(ba)}")

print("\n✅ All types working!")
EOF
```

### 3. Test Individual Type

```bash
# Test PySet
python3 -c "s = {1,2,3}; s.add(4); print('Set:', s)"

# Test PyComplex
python3 -c "print('Complex abs:', abs(complex(3, 4)))"

# Test PyDecimal
python3 -c "from decimal import Decimal; print('Decimal:', Decimal('0.1') + Decimal('0.2'))"

# Test PyDateTime
python3 -c "from datetime import datetime; print('Now:', datetime.now())"

# Test PyPath
python3 -c "from pathlib import Path; print('CWD:', Path.cwd())"

# Test PyUUID
python3 -c "import uuid; print('UUID4:', uuid.uuid4())"
```

## What Gets Tested

### 1. Zig Unit Tests
- Type wrapper compilation
- FFI bindings
- Export correctness

### 2. Python Type Compatibility Tests
- ✅ **PyComplex**: Complex arithmetic, abs, conjugate
- ✅ **PyDecimal**: Precise arithmetic (0.1 + 0.2 = 0.3)
- ✅ **PyDateTime**: datetime, date, time, timedelta
- ✅ **PyPath**: File I/O, exists, read/write
- ✅ **PyUUID**: uuid4, uuid5, namespace
- ✅ **PySet/PyFrozenSet**: Union, intersection, membership
- ✅ **PyRange**: Length, membership, iteration
- ✅ **PyByteArray**: Mutation, extend, reverse
- ✅ **PyGenerator**: Iteration, protocol

### 3. Integration Tests
Tests multiple types working together:
- Financial report with UUID, DateTime, Decimal
- File operations with Path
- Set operations with collections
- Date calculations with timedelta

## Troubleshooting

### Zig Not Found
```bash
# Install Zig 0.14.0 or later
brew install zig  # macOS
# or download from https://ziglang.org/
```

### Python Not Found
```bash
# Ensure Python 3.11+ is installed
python3 --version
```

### Build Fails
```bash
# Clean and rebuild
./test_new_types.sh --clean

# Or manually
rm -rf zig-out zig-cache
zig build
```

### pytest Not Found
```bash
# Install pytest
python3 -m pip install pytest --user
```

### Import Errors
```bash
# Make sure the build succeeded
ls -la zig-out/lib/

# Check for .so or .dylib files
file zig-out/lib/*
```

## Test Output

### Successful Run
```
========================================
Checking Prerequisites
========================================

✅ Zig found: 0.14.0
✅ Python found: Python 3.11.5
✅ pytest found

========================================
Building Ziggy pyZ3
========================================

ℹ️  Running: zig build
✅ Main project built successfully

========================================
Running Python Type Compatibility
========================================

✅ PyComplex: complex(3, 4) works correctly
✅ PyDecimal: 0.1 + 0.2 = 0.3 (exact)
✅ PyDateTime/PyDate/PyTime/PyTimeDelta: all working
✅ PyPath: file operations working
✅ PyUUID: uuid4 and uuid5 working
✅ PySet/PyFrozenSet: set operations working
✅ PyRange: range operations working
✅ PyByteArray: mutable operations working
✅ PyGenerator: generator protocol working

Results: 9 passed, 0 failed

✅ All Python type tests passed

========================================
Running Integration Test
========================================

✅ Integration test passed!

Report ID: 12345678-1234-5678-1234-567812345678
Timestamp: 2025-12-04T10:30:00.123456
Total: $64.78
Due Date: 2026-01-03T10:30:00.123456
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Test New Types
  run: |
    chmod +x test_new_types.sh
    ./test_new_types.sh --all
```

### Local Pre-commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
./test_new_types.sh --quick || exit 1
```

## Performance

- **Quick verification**: ~1 second
- **Python type tests**: ~2-3 seconds
- **Full test suite**: ~10-15 seconds (includes build)
- **Clean rebuild**: ~30-60 seconds

## Files Created by Script

Temporary test files:
- `/tmp/test_types.py` - Python type compatibility tests
- `/tmp/integration_test.py` - Integration test
- `/tmp/zig_test.log` - Zig test output log

These are automatically cleaned up on each run.

## Next Steps

After successful testing:

1. ✅ All types verified locally
2. ✅ Build succeeds
3. ✅ Tests pass
4. → Ready to use in your Python extensions!

## Example Usage in Your Code

```zig
const py = @import("pyz3");

pub fn example() !py.PyObject {
    // Use PySet
    const my_set = try py.PySet(root).new();

    // Use PyComplex
    const c = try py.PyComplex.create(3.0, 4.0);

    // Use PyDecimal
    const d = try py.PyDecimal.fromString("99.99");

    // Use PyDateTime
    const now = try py.PyDateTime.now();

    // Use PyPath
    const path = try py.PyPath.cwd();

    // Use PyUUID
    const uuid = try py.PyUUID.uuid4();

    return my_set.obj;
}
```

## Support

If you encounter issues:

1. Check the error output carefully
2. Run `./test_new_types.sh --quick` first
3. Try `./test_new_types.sh --clean` to rebuild
4. Check Zig and Python versions
5. Review build logs in `/tmp/zig_test.log`

## Summary

The test script provides:
- ✅ Automated testing of all 13 new types
- ✅ Interactive menu for selective testing
- ✅ Quick verification for CI/CD
- ✅ Detailed error reporting
- ✅ Integration testing
- ✅ Clean build management

Happy testing! 🎉
