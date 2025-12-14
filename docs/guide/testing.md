# Testing Guide for pyz3

This guide covers the testing infrastructure, helpers, and best practices for writing tests in the pyz3 project.

## Table of Contents

1. [Test Infrastructure](#test-infrastructure)
2. [Test Helpers](#test-helpers)
3. [Writing Tests](#writing-tests)
4. [Performance Testing](#performance-testing)
5. [Best Practices](#best-practices)

## Test Infrastructure

### Directory Structure

```
test/
├── conftest.py           # Pytest configuration and fixtures
├── helpers.py            # Test helper utilities
├── test_*.py             # Test files
├── test_performance.py   # Performance regression tests
└── test_helpers_demo.py  # Helper usage examples
```

### Running Tests

```bash
# Run all tests
./run_all_tests.sh

# Run specific test file
poetry run pytest test/test_hello.py -v

# Run with coverage
poetry run pytest test/ --cov=pyz3

# Run only fast tests (exclude slow/benchmark tests)
poetry run pytest test/ -m "not slow and not benchmark"

# Run performance benchmarks
poetry run pytest test/test_performance.py -v
```

## Test Helpers

### Available Helper Classes

The `test/helpers.py` module provides standardized utilities:

#### 1. ExceptionTester

Helper for testing exception raising behavior.

```python
from test.helpers import ExceptionTester

# Assert exception is raised
exc = ExceptionTester.assert_raises(ValueError, int, "not a number")

# Assert exception with specific message
exc = ExceptionTester.assert_raises_with_message(
    ValueError, "invalid literal", int, "not a number"
)

# Context manager for exception testing
with ExceptionTester.expect_exception(ValueError, match="invalid"):
    raise ValueError("invalid input")
```

#### 2. TypeChecker

Helpers for checking Python types and signatures.

```python
from test.helpers import TypeChecker
import inspect

# Check function signature
TypeChecker.assert_signature(
    my_function,
    [inspect.Parameter("x", inspect.Parameter.POSITIONAL_ONLY)]
)

# Check return type annotation
TypeChecker.assert_return_type(my_function, int)

# Check instance type
assert TypeChecker.check_type(value, str)
```

#### 3. PerformanceTester

Helpers for performance testing and benchmarking.

```python
from test.helpers import PerformanceTester

perf = PerformanceTester()

# Benchmark a function
results = perf.benchmark(my_function, arg1, arg2, iterations=1000)
print(f"Mean: {results['mean_ms']:.4f}ms")
print(f"P95: {results['p95_ms']:.4f}ms")

# Compare multiple implementations
results = perf.compare({
    'implementation_a': lambda: func_a(x),
    'implementation_b': lambda: func_b(x),
}, iterations=1000)

# Time a single function call
@PerformanceTester.time_function
def my_operation():
    return expensive_calc()

result, elapsed_ms = my_operation()
```

#### 4. MemoryTester

Helpers for memory leak detection.

```python
from test.helpers import MemoryTester

# Check for memory leaks
with MemoryTester.check_no_leaks(tolerance=5):
    for _ in range(1000):
        result = my_function()
        del result

# Get reference count
refcount = MemoryTester.get_refcount(obj)

# Get total object count
count = MemoryTester.get_object_count()
```

#### 5. DataGenerator

Helpers for generating test data.

```python
from test.helpers import DataGenerator

# Generate numbers
numbers = DataGenerator.numbers(10, start=0, step=2)
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Generate floats
floats = DataGenerator.floats(5, start=1.0, step=0.5)
# [1.0, 1.5, 2.0, 2.5, 3.0]

# Generate strings
strings = DataGenerator.strings(3, prefix="test")
# ['test_0', 'test_1', 'test_2']

# Generate nested structures
nested = DataGenerator.nested_list(depth=2, width=3, value=0)
# [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
```

### Module Loading

```python
from test.helpers import get_example_module

# Lazy load and cache modules
hello = get_example_module('hello')
functions = get_example_module('functions')
```

## Writing Tests

### Using Fixtures

The `conftest.py` provides several useful fixtures:

```python
def test_with_fixtures(hello_module, functions_module, perf):
    """Example using provided fixtures."""
    # hello_module is pre-imported
    assert hello_module.hello() == "Hello!"

    # functions_module is pre-imported
    result = functions_module.double(21)
    assert result == 42

    # perf is a PerformanceTester instance
    benchmark = perf.benchmark(functions_module.double, 42)
    assert benchmark['mean_ms'] < 1.0
```

Available fixtures:
- `example` - Main example module with submodules
- `hello_module` - Hello example module
- `functions_module` - Functions example module
- `memory_module` - Memory example module
- `exceptions_module` - Exceptions example module
- `perf` - PerformanceTester instance
- `perf_baseline` - Performance baseline tracker
- `project_root` - Path to project root directory

### Test Organization

Organize tests by feature/module:

```python
class TestFeatureName:
    """Test suite for specific feature."""

    def test_basic_behavior(self):
        """Test basic functionality."""
        pass

    def test_edge_cases(self):
        """Test edge cases."""
        pass

    def test_error_handling(self):
        """Test error handling."""
        pass
```

### Parameterized Tests

Use the `parametrize_with_ids` helper for cleaner parameterization:

```python
from test.helpers import parametrize_with_ids

@parametrize_with_ids(
    'input_val',
    [1, 10, 100, 1000],
    ids=['small', 'medium', 'large', 'xlarge']
)
def test_scaling(input_val):
    """Test behavior at different scales."""
    result = my_function(input_val)
    assert result > 0
```

## Performance Testing

### Writing Performance Tests

```python
import pytest
from test.helpers import PerformanceTester

class TestPerformance:
    """Performance test suite."""

    def test_operation_speed(self, perf_baseline):
        """Test operation meets performance baseline."""
        perf = PerformanceTester()

        results = perf.benchmark(my_operation, iterations=1000)

        # Check against baseline (20% tolerance)
        assert perf_baseline.check_regression(
            'my_operation_mean_ms',
            results['mean_ms'],
            tolerance=1.2
        ), f"Performance regression: {results['mean_ms']:.4f}ms"

    @pytest.mark.slow
    @pytest.mark.parametrize('size', [10, 100, 1000, 10000])
    def test_scalability(self, size, perf_baseline):
        """Test scalability with different input sizes."""
        perf = PerformanceTester()

        data = list(range(size))
        results = perf.benchmark(my_operation, data, iterations=100)

        # Per-element time should be constant
        per_element_ms = results['mean_ms'] / size

        key = f'my_operation_per_element_{size}'
        assert perf_baseline.check_regression(key, per_element_ms, tolerance=1.5)
```

### Performance Markers

- `@pytest.mark.slow` - Slow tests (skipped in quick runs)
- `@pytest.mark.benchmark` - Benchmark tests (detailed performance tracking)

### Baseline Tracking

The `perf_baseline` fixture automatically tracks and compares against performance baselines:

```python
def test_with_baseline(perf_baseline):
    # First run: establishes baseline
    perf_baseline.set('key', 1.234)

    # Subsequent runs: check regression
    current = 1.456
    assert perf_baseline.check_regression('key', current, tolerance=1.2)
```

## Best Practices

### 1. Use Fixtures for Common Setup

```python
# Good
def test_with_fixture(functions_module):
    result = functions_module.double(42)
    assert result == 84

# Avoid
def test_without_fixture():
    from example import functions
    result = functions.double(42)
    assert result == 84
```

### 2. Use Helpers for Common Patterns

```python
# Good
ExceptionTester.assert_raises_with_message(
    ValueError, "expected message", my_func, arg
)

# Avoid
with pytest.raises(ValueError) as exc:
    my_func(arg)
assert str(exc.value) == "expected message"
```

### 3. Test Performance Critical Paths

```python
@pytest.mark.benchmark
def test_critical_operation(perf, perf_baseline):
    """Benchmark critical operation."""
    results = perf.benchmark(critical_func, iterations=10000)
    assert perf_baseline.check_regression(
        'critical_func_mean_ms',
        results['mean_ms'],
        tolerance=1.1  # Strict 10% tolerance
    )
```

### 4. Check for Memory Leaks

```python
def test_no_leaks():
    """Verify no memory leaks."""
    with MemoryTester.check_no_leaks(tolerance=5):
        for _ in range(1000):
            result = my_function()
            del result
```

### 5. Use Descriptive Test Names

```python
# Good
def test_double_function_returns_twice_input_value():
    pass

# Avoid
def test_double():
    pass
```

### 6. Document Test Intent

```python
def test_division_by_zero_raises_error():
    """
    Test that dividing by zero raises ZeroDivisionError.

    This is critical because the C implementation must properly
    handle this edge case and convert to Python exception.
    """
    ExceptionTester.assert_raises(ZeroDivisionError, divide, 10, 0)
```

### 7. Organize with Test Classes

```python
class TestArithmeticOperations:
    """Test suite for arithmetic operations."""

    def test_addition(self):
        """Test addition operator."""
        pass

    def test_subtraction(self):
        """Test subtraction operator."""
        pass
```

## Examples

### Complete Test Example

```python
"""Tests for my_module functionality."""

import pytest
import inspect
from test.helpers import (
    ExceptionTester,
    TypeChecker,
    PerformanceTester,
    get_example_module,
    parametrize_with_ids,
)


class TestMyModule:
    """Test suite for my_module."""

    @pytest.fixture
    def module(self):
        """Load module fixture."""
        return get_example_module('my_module')

    def test_basic_functionality(self, module):
        """Test basic function works correctly."""
        result = module.my_function(42)
        assert result == 84

    def test_type_validation(self, module):
        """Test function validates input types."""
        ExceptionTester.assert_raises(
            TypeError,
            module.my_function,
            "not an int",
            match="expected int"
        )

    def test_signature(self, module):
        """Test function has correct signature."""
        TypeChecker.assert_signature(
            module.my_function,
            [inspect.Parameter("x", inspect.Parameter.POSITIONAL_ONLY)]
        )

    @pytest.mark.benchmark
    def test_performance(self, module, perf, perf_baseline):
        """Test performance meets baseline."""
        results = perf.benchmark(module.my_function, 42, iterations=10000)

        assert perf_baseline.check_regression(
            'my_function_mean_ms',
            results['mean_ms'],
            tolerance=1.2
        )

        # Should be fast
        assert results['mean_ms'] < 0.1

    def test_no_memory_leaks(self, module):
        """Verify no memory leaks."""
        with MemoryTester.check_no_leaks(tolerance=5):
            for _ in range(1000):
                result = module.my_function(42)
                del result

    @parametrize_with_ids(
        'value',
        [0, 1, 10, 100, 1000],
        ids=['zero', 'one', 'ten', 'hundred', 'thousand']
    )
    def test_various_inputs(self, module, value):
        """Test with various input values."""
        result = module.my_function(value)
        assert result == value * 2
```

## Troubleshooting

### Test Fails to Import Module

Ensure module is registered in `pyproject.toml`:

```toml
[[tool.pyz3.ext_module]]
name = "example.my_module"
root = "example/my_module.zig"
```

### Performance Test Fails

1. Check if baseline was established
2. Verify tolerance is reasonable (1.2 = 20% regression allowed)
3. Run multiple times to eliminate noise
4. Consider system load during testing

### Memory Leak Test Fails

1. Increase tolerance if legitimate objects are created
2. Check if garbage collection is running
3. Verify proper cleanup (del statements, context managers)

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [Contributing Guide](../../CONTRIBUTING.md)
- [Architecture Documentation](../../ARCHITECTURE.md)
