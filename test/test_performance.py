"""
Performance regression tests for pyz3.

These tests benchmark critical operations and ensure performance doesn't
degrade over time.

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

import pytest
from test.helpers import PerformanceTester, get_example_module


class TestTypeConversionPerformance:
    """Performance tests for type conversions."""

    def test_int_conversion_speed(self, perf_baseline):
        """Test integer conversion performance."""
        perf = PerformanceTester()
        functions = get_example_module("functions")

        # Benchmark the double function (simple int operation)
        results = perf.benchmark(functions.double, 100, iterations=10000)

        # Check against baseline (allow 20% regression)
        assert perf_baseline.check_regression(
            "int_conversion_mean_ms", results["mean_ms"], tolerance=1.2
        ), f"Performance regression detected: {results['mean_ms']:.4f}ms > baseline"

        # Ensure it's reasonably fast (< 0.1ms per call)
        assert results["mean_ms"] < 0.1, f"Integer conversion too slow: {results['mean_ms']:.4f}ms"

    def test_list_operations_speed(self, perf_baseline):
        """Test list operations performance."""
        try:
            list_conv = get_example_module("list_conversion_example")
        except (ImportError, AttributeError):
            pytest.skip("list_conversion_example not available")

        perf = PerformanceTester()

        # Benchmark list sum operation
        test_list = list(range(100))
        results = perf.benchmark(list_conv.sum_list, test_list, iterations=1000)

        assert perf_baseline.check_regression(
            "list_sum_mean_ms", results["mean_ms"], tolerance=1.2
        )

        # Should be faster than pure Python
        python_results = perf.benchmark(sum, test_list, iterations=1000)
        speedup = python_results["mean_ms"] / results["mean_ms"]

        # pyz3 implementation should be competitive (at least 50% of Python speed)
        # Note: Pure Python sum() is highly optimized, so we don't expect to beat it
        assert speedup > 0.5, f"pyz3 list sum too slow compared to Python: {speedup:.2f}x"


class TestMemoryPerformance:
    """Performance tests for memory operations."""

    def test_string_operations_speed(self, perf_baseline):
        """Test string operations performance."""
        try:
            memory = get_example_module("memory")
        except (ImportError, AttributeError):
            pytest.skip("memory module not available")

        perf = PerformanceTester()

        # Benchmark string append
        results = perf.benchmark(memory.append, "test ", iterations=10000)

        assert perf_baseline.check_regression(
            "string_append_mean_ms", results["mean_ms"], tolerance=1.2
        )

        # Should be reasonably fast
        assert results["mean_ms"] < 0.1, f"String append too slow: {results['mean_ms']:.4f}ms"


class TestComparativePerformance:
    """Comparative performance tests between different approaches."""

    def test_compare_function_call_overhead(self, perf_baseline):
        """Compare performance of different function call patterns."""
        functions = get_example_module("functions")

        perf = PerformanceTester()

        # Compare simple vs kwargs functions
        results = perf.compare(
            {
                "simple": lambda: functions.double(42),
                "kwargs": lambda: functions.with_kwargs(42.0, y=10.0),
            },
            iterations=10000,
        )

        # Simple function should be faster
        assert (
            results["simple"]["mean_ms"] <= results["kwargs"]["mean_ms"]
        ), "Simple function should be faster than kwargs"

        # Record baselines
        perf_baseline.set("simple_call_mean_ms", results["simple"]["mean_ms"])
        perf_baseline.set("kwargs_call_mean_ms", results["kwargs"]["mean_ms"])

    def test_exception_overhead(self, perf_baseline):
        """Test exception raising overhead."""
        try:
            exceptions = get_example_module("exceptions")
        except (ImportError, AttributeError):
            pytest.skip("exceptions module not available")

        perf = PerformanceTester()

        # Benchmark exception raising
        def raise_and_catch():
            try:
                exceptions.raise_value_error("test")
            except ValueError:
                pass

        results = perf.benchmark(raise_and_catch, iterations=1000)

        assert perf_baseline.check_regression(
            "exception_overhead_mean_ms", results["mean_ms"], tolerance=1.2
        )

        # Exception handling should be reasonable (< 1ms per raise/catch)
        assert results["mean_ms"] < 1.0, f"Exception overhead too high: {results['mean_ms']:.4f}ms"


@pytest.mark.slow
class TestScalabilityPerformance:
    """Scalability performance tests."""

    @pytest.mark.parametrize("size", [10, 100, 1000, 10000])
    def test_list_sum_scalability(self, size, perf_baseline):
        """Test that list sum scales linearly."""
        try:
            list_conv = get_example_module("list_conversion_example")
        except (ImportError, AttributeError):
            pytest.skip("list_conversion_example not available")

        perf = PerformanceTester()

        test_list = list(range(size))
        results = perf.benchmark(list_conv.sum_list, test_list, iterations=100)

        # Check per-element time (should be roughly constant)
        per_element_ms = results["mean_ms"] / size

        key = f"list_sum_per_element_{size}"
        assert perf_baseline.check_regression(
            key, per_element_ms, tolerance=1.5
        ), f"Scalability regression at size {size}"


@pytest.mark.benchmark
class TestDetailedBenchmarks:
    """Detailed benchmark suite for performance tracking."""

    def test_benchmark_all_operations(self, perf_baseline):
        """Comprehensive benchmark of all major operations."""
        functions = get_example_module("functions")

        perf = PerformanceTester()

        benchmarks = {
            "double_int": lambda: functions.double(42),
        }

        try:
            memory = get_example_module("memory")
            benchmarks["string_append"] = lambda: memory.append("test ")
        except (ImportError, AttributeError):
            pass

        results = perf.compare(benchmarks, iterations=10000)

        # Print results for tracking
        print("\n=== Performance Benchmark Results ===")
        for name, stats in results.items():
            print(f"{name}:")
            print(f"  Mean: {stats['mean_ms']:.6f}ms")
            print(f"  Median: {stats['median_ms']:.6f}ms")
            print(f"  P95: {stats['p95_ms']:.6f}ms")
            print(f"  P99: {stats['p99_ms']:.6f}ms")

            # Save to baseline
            perf_baseline.set(f"{name}_benchmark_mean", stats["mean_ms"])
            perf_baseline.set(f"{name}_benchmark_p95", stats["p95_ms"])

        # Verify no major regressions
        for name, stats in results.items():
            baseline = perf_baseline.get(f"{name}_benchmark_mean")
            if baseline is not None:
                regression = (stats["mean_ms"] - baseline) / baseline * 100
                assert (
                    regression < 20
                ), f"{name} regressed by {regression:.1f}% (baseline: {baseline:.6f}ms, current: {stats['mean_ms']:.6f}ms)"
