"""
Demonstration of test helpers usage.

This file shows how to use the standardized test helpers for cleaner,
more maintainable tests.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import inspect

import pytest
from test.helpers import (
    ExceptionTester,
    TypeChecker,
    PerformanceTester,
    MemoryTester,
    DataGenerator,
    get_example_module,
    parametrize_with_ids,
)


class TestExceptionHelpers:
    """Demonstrate exception testing helpers."""

    def test_exception_with_helper(self):
        """Using ExceptionTester for cleaner exception testing."""
        exceptions = get_example_module("exceptions")

        # Old way:
        # with pytest.raises(ValueError) as exc:
        #     exceptions.raise_value_error("hello!")
        # assert str(exc.value) == "hello!"

        # New way with helper:
        exc = ExceptionTester.assert_raises_with_message(
            ValueError, "hello!", exceptions.raise_value_error, "hello!"
        )
        assert isinstance(exc, ValueError)

    def test_exception_context_manager(self):
        """Using context manager for exception testing."""
        exceptions = get_example_module("exceptions")

        with ExceptionTester.expect_exception(ValueError, match="hello"):
            exceptions.raise_value_error("hello!")


class TestTypeCheckingHelpers:
    """Demonstrate type checking helpers."""

    def test_signature_checking(self):
        """Using TypeChecker for signature verification."""
        functions = get_example_module("functions")

        # Verify function signature with helper
        TypeChecker.assert_signature(
            functions.double,
            [inspect.Parameter("x", kind=inspect.Parameter.POSITIONAL_ONLY)],
        )

    def test_type_verification(self):
        """Using TypeChecker for type verification."""
        hello = get_example_module("hello")

        result = hello.hello()
        assert TypeChecker.check_type(result, str)


class TestPerformanceHelpers:
    """Demonstrate performance testing helpers."""

    def test_simple_benchmark(self):
        """Using PerformanceTester for benchmarking."""
        functions = get_example_module("functions")

        perf = PerformanceTester()
        results = perf.benchmark(functions.double, 42, iterations=1000)

        # Verify benchmark structure
        assert "mean_ms" in results
        assert "median_ms" in results
        assert "p95_ms" in results

        # Basic performance assertion
        assert results["mean_ms"] < 1.0, "Function should be fast"

    def test_comparative_benchmark(self):
        """Compare performance of different implementations."""
        functions = get_example_module("functions")

        perf = PerformanceTester()

        # Compare Python implementation vs Zig implementation
        results = perf.compare(
            {
                "zig": lambda: functions.double(42),
                "python": lambda: 42 * 2,
            },
            iterations=10000,
        )

        # Both should complete
        assert results["zig"]["mean_ms"] > 0
        assert results["python"]["mean_ms"] > 0

        # Speedup should be calculated
        assert "speedup" in results["zig"]
        assert "speedup" in results["python"]

    def test_timed_function_decorator(self):
        """Using time_function decorator."""
        functions = get_example_module("functions")

        @PerformanceTester.time_function
        def test_op():
            return functions.double(42)

        result, elapsed_ms = test_op()

        assert result == 84
        assert elapsed_ms >= 0


class TestMemoryHelpers:
    """Demonstrate memory testing helpers."""

    def test_no_leak_check(self):
        """Using MemoryTester to check for leaks."""
        hello = get_example_module("hello")

        # This should not leak
        with MemoryTester.check_no_leaks(tolerance=10):
            for _ in range(100):
                result = hello.hello()
                del result

    def test_refcount_tracking(self):
        """Track reference counts."""
        obj = [1, 2, 3]

        # Get initial refcount
        initial = MemoryTester.get_refcount(obj)

        # Create another reference
        obj2 = obj

        # Refcount should increase
        assert MemoryTester.get_refcount(obj) == initial + 1

        # Clean up
        del obj2


class TestDataGenerationHelpers:
    """Demonstrate data generation helpers."""

    def test_number_generation(self):
        """Generate test data with DataGenerator."""
        numbers = DataGenerator.numbers(10, start=5, step=2)
        assert numbers == [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

    def test_float_generation(self):
        """Generate float test data."""
        floats = DataGenerator.floats(5, start=1.0, step=0.5)
        assert floats == [1.0, 1.5, 2.0, 2.5, 3.0]

    def test_string_generation(self):
        """Generate string test data."""
        strings = DataGenerator.strings(3, prefix="item")
        assert strings == ["item_0", "item_1", "item_2"]

    def test_nested_list_generation(self):
        """Generate nested list structures."""
        nested = DataGenerator.nested_list(depth=2, width=3, value=0)
        assert nested == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


class TestModuleLoadingHelpers:
    """Demonstrate module loading helpers."""

    def test_lazy_module_loading(self):
        """Using get_example_module for lazy loading."""
        # Load module only when needed
        hello = get_example_module("hello")
        assert hello.hello() == "Hello!"

        # Same module is cached
        hello2 = get_example_module("hello")
        assert hello is hello2


class TestParametrizeHelpers:
    """Demonstrate parametrize helpers."""

    @parametrize_with_ids(
        "value", [1, 10, 100], ids=["small", "medium", "large"]
    )
    def test_with_named_params(self, value):
        """Test with named parameters for better test IDs."""
        functions = get_example_module("functions")
        result = functions.double(value)
        assert result == value * 2


class TestFixtureIntegration:
    """Demonstrate integration with new fixtures."""

    def test_using_hello_fixture(self, hello_module):
        """Use hello_module fixture instead of importing."""
        assert hello_module.hello() == "Hello!"

    def test_using_functions_fixture(self, functions_module):
        """Use functions_module fixture."""
        assert functions_module.double(21) == 42

    def test_using_perf_fixture(self, perf):
        """Use perf fixture for benchmarking."""
        functions = get_example_module("functions")

        results = perf.benchmark(functions.double, 42, iterations=100)
        assert results["mean_ms"] >= 0


# Example of a complete refactored test class
class TestCompleteExample:
    """
    Complete example showing best practices with all helpers.

    This demonstrates how to write clean, maintainable tests using
    the standardized helper utilities.
    """

    def test_function_behavior(self, functions_module):
        """Test basic function behavior with fixtures."""
        result = functions_module.double(21)
        assert result == 42

    def test_exception_handling(self, functions_module):
        """Test exception handling with helpers."""
        ExceptionTester.assert_raises(
            TypeError,
            functions_module.double,
            0.1,  # Should be int, not float
            match="expected int",
        )

    def test_signature_verification(self, functions_module):
        """Verify function signatures with TypeChecker."""
        TypeChecker.assert_signature(
            functions_module.double,
            [inspect.Parameter("x", kind=inspect.Parameter.POSITIONAL_ONLY)],
        )

    @pytest.mark.benchmark
    def test_performance_baseline(self, functions_module, perf, perf_baseline):
        """Track performance against baseline."""
        results = perf.benchmark(functions_module.double, 42, iterations=1000)

        # Check against baseline with 20% tolerance
        assert perf_baseline.check_regression(
            "demo_double_mean_ms", results["mean_ms"], tolerance=1.2
        ), f"Performance regression: {results['mean_ms']:.4f}ms"

    @parametrize_with_ids("input_val", [0, 1, 10, 100], ids=["zero", "one", "ten", "hundred"])
    def test_parameterized_values(self, functions_module, input_val):
        """Test with multiple input values."""
        result = functions_module.double(input_val)
        assert result == input_val * 2

    def test_no_memory_leaks(self, functions_module):
        """Verify no memory leaks during operations."""
        with MemoryTester.check_no_leaks(tolerance=5):
            for _ in range(1000):
                result = functions_module.double(42)
                del result
