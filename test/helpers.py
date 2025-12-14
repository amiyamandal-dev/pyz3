"""
Test helpers and utilities for pyz3 test suite.

This module provides common testing utilities, fixtures, and assertion helpers
to standardize and simplify testing across the pyz3 project.

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

import contextlib
import functools
import gc
import inspect
import sys
import time
from typing import Any, Callable, ContextManager, Dict, List, Optional, Type, TypeVar

import pytest

T = TypeVar("T")


# =============================================================================
# Module Import Helpers
# =============================================================================


class LazyModuleLoader:
    """Lazy module loader that imports only once and caches."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def load(self, module_name: str):
        """Load and cache a module."""
        if module_name not in self._cache:
            parts = module_name.split(".")
            module = __import__(module_name)
            for part in parts[1:]:
                module = getattr(module, part)
            self._cache[module_name] = module
        return self._cache[module_name]

    def clear(self):
        """Clear module cache."""
        self._cache.clear()


# Global lazy loader instance
_module_loader = LazyModuleLoader()


def get_example_module(name: str):
    """
    Get an example module by name.

    Args:
        name: Module name relative to 'example' (e.g., 'hello', 'functions')

    Returns:
        The imported module

    Example:
        >>> hello = get_example_module('hello')
        >>> assert hello.hello() == "Hello!"
    """
    return _module_loader.load(f"example.{name}")


# =============================================================================
# Exception Testing Helpers
# =============================================================================


class ExceptionTester:
    """Helper for testing exception raising behavior."""

    @staticmethod
    def assert_raises(
        exc_type: Type[Exception],
        func: Callable,
        *args,
        match: Optional[str] = None,
        **kwargs,
    ) -> Exception:
        """
        Assert that a function raises a specific exception.

        Args:
            exc_type: Expected exception type
            func: Function to call
            *args: Positional arguments for func
            match: Optional regex pattern to match exception message
            **kwargs: Keyword arguments for func

        Returns:
            The caught exception

        Example:
            >>> ExceptionTester.assert_raises(ValueError, int, "not a number")
        """
        with pytest.raises(exc_type, match=match) as exc_info:
            func(*args, **kwargs)
        return exc_info.value

    @staticmethod
    def assert_raises_with_message(
        exc_type: Type[Exception], expected_msg: str, func: Callable, *args, **kwargs
    ) -> Exception:
        """Assert exception is raised with exact message."""
        exc = ExceptionTester.assert_raises(exc_type, func, *args, **kwargs)
        assert str(exc) == expected_msg, f"Expected '{expected_msg}', got '{str(exc)}'"
        return exc

    @staticmethod
    @contextlib.contextmanager
    def expect_exception(exc_type: Type[Exception], match: Optional[str] = None):
        """
        Context manager for expecting an exception.

        Example:
            >>> with ExceptionTester.expect_exception(ValueError, match="invalid"):
            ...     raise ValueError("invalid input")
        """
        with pytest.raises(exc_type, match=match):
            yield


# =============================================================================
# Type Checking Helpers
# =============================================================================


class TypeChecker:
    """Helpers for checking Python types and signatures."""

    @staticmethod
    def assert_signature(
        func: Callable, expected_params: List[inspect.Parameter]
    ) -> None:
        """
        Assert function has expected signature.

        Args:
            func: Function to check
            expected_params: List of expected parameters

        Example:
            >>> def foo(x: int, y: str = "default"): pass
            >>> TypeChecker.assert_signature(foo, [
            ...     inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            ...     inspect.Parameter("y", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="default", annotation=str)
            ... ])
        """
        sig = inspect.signature(func)
        expected_sig = inspect.Signature(expected_params)
        assert sig == expected_sig, f"Expected {expected_sig}, got {sig}"

    @staticmethod
    def assert_return_type(func: Callable, expected_type: Type) -> None:
        """Assert function has expected return type annotation."""
        sig = inspect.signature(func)
        assert (
            sig.return_annotation == expected_type
        ), f"Expected return type {expected_type}, got {sig.return_annotation}"

    @staticmethod
    def check_type(value: Any, expected_type: Type) -> bool:
        """Check if value is of expected type."""
        return isinstance(value, expected_type)


# =============================================================================
# Performance Testing Helpers
# =============================================================================


class PerformanceTester:
    """Helpers for performance testing and benchmarking."""

    def __init__(self):
        self.results: Dict[str, List[float]] = {}

    def benchmark(
        self,
        func: Callable[..., T],
        *args,
        iterations: int = 1000,
        warmup: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Benchmark a function.

        Args:
            func: Function to benchmark
            *args: Arguments to pass to func
            iterations: Number of iterations to run
            warmup: Number of warmup iterations
            **kwargs: Keyword arguments to pass to func

        Returns:
            Dict with timing statistics

        Example:
            >>> perf = PerformanceTester()
            >>> results = perf.benchmark(lambda: sum(range(1000)), iterations=1000)
            >>> assert results['mean_ms'] < 1.0  # Should be fast
        """
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds

        times_sorted = sorted(times)
        return {
            "iterations": iterations,
            "mean_ms": sum(times) / len(times),
            "median_ms": times_sorted[len(times_sorted) // 2],
            "min_ms": min(times),
            "max_ms": max(times),
            "p95_ms": times_sorted[int(len(times_sorted) * 0.95)],
            "p99_ms": times_sorted[int(len(times_sorted) * 0.99)],
        }

    def compare(
        self,
        funcs: Dict[str, Callable],
        *args,
        iterations: int = 1000,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare performance of multiple functions.

        Args:
            funcs: Dict mapping names to functions
            *args: Arguments to pass to all functions
            iterations: Number of iterations
            **kwargs: Keyword arguments to pass to all functions

        Returns:
            Dict mapping names to benchmark results

        Example:
            >>> perf = PerformanceTester()
            >>> results = perf.compare({
            ...     'python': lambda x: sum(range(x)),
            ...     'builtin': lambda x: sum(range(x)),
            ... }, 1000)
        """
        results = {}
        for name, func in funcs.items():
            results[name] = self.benchmark(func, *args, iterations=iterations, **kwargs)

        # Add speedup comparisons
        if len(results) > 1:
            baseline = list(results.values())[0]["mean_ms"]
            for name, result in results.items():
                result["speedup"] = baseline / result["mean_ms"]

        return results

    @staticmethod
    def time_function(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
        """
        Decorator that times function execution.

        Returns:
            Tuple of (result, execution_time_ms)

        Example:
            >>> @PerformanceTester.time_function
            ... def slow_function():
            ...     time.sleep(0.1)
            ...     return 42
            >>> result, elapsed = slow_function()
            >>> assert result == 42
            >>> assert elapsed >= 100  # milliseconds
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            return result, elapsed_ms

        return wrapper


# =============================================================================
# Memory Testing Helpers
# =============================================================================


class MemoryTester:
    """Helpers for memory leak detection and testing."""

    @staticmethod
    @contextlib.contextmanager
    def check_no_leaks(tolerance: int = 0):
        """
        Context manager to check for memory leaks.

        Args:
            tolerance: Number of allowed leaked objects

        Example:
            >>> with MemoryTester.check_no_leaks():
            ...     # Code that should not leak
            ...     x = [1, 2, 3]
            ...     del x
        """
        gc.collect()
        before = len(gc.get_objects())

        yield

        gc.collect()
        after = len(gc.get_objects())

        leaked = after - before
        assert (
            leaked <= tolerance
        ), f"Memory leak detected: {leaked} objects leaked (tolerance: {tolerance})"

    @staticmethod
    def get_object_count() -> int:
        """Get current number of tracked objects."""
        gc.collect()
        return len(gc.get_objects())

    @staticmethod
    def get_refcount(obj: Any) -> int:
        """Get reference count for an object."""
        return sys.getrefcount(obj) - 1  # Subtract 1 for the getrefcount call itself


# =============================================================================
# Data Generation Helpers
# =============================================================================


class DataGenerator:
    """Helpers for generating test data."""

    @staticmethod
    def numbers(count: int, start: int = 0, step: int = 1) -> List[int]:
        """Generate a list of numbers."""
        return list(range(start, start + count * step, step))

    @staticmethod
    def floats(count: int, start: float = 0.0, step: float = 1.0) -> List[float]:
        """Generate a list of floats."""
        return [start + i * step for i in range(count)]

    @staticmethod
    def strings(count: int, prefix: str = "test") -> List[str]:
        """Generate a list of strings."""
        return [f"{prefix}_{i}" for i in range(count)]

    @staticmethod
    def nested_list(depth: int, width: int, value: Any = 0) -> List:
        """Generate a nested list structure."""
        if depth == 1:
            return [value] * width
        return [DataGenerator.nested_list(depth - 1, width, value) for _ in range(width)]


# =============================================================================
# Global Instances
# =============================================================================

# Export singleton instances for convenience
exc_tester = ExceptionTester()
type_checker = TypeChecker()
mem_tester = MemoryTester()
data_gen = DataGenerator()


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def perf():
    """Fixture providing PerformanceTester instance."""
    return PerformanceTester()


@pytest.fixture
def module_loader():
    """Fixture providing LazyModuleLoader instance."""
    loader = LazyModuleLoader()
    yield loader
    loader.clear()


# =============================================================================
# Utility Functions
# =============================================================================


def skip_if_no_module(module_name: str):
    """Skip test if module is not available."""

    def decorator(func):
        try:
            __import__(module_name)
            return func
        except ImportError:
            return pytest.mark.skip(reason=f"Module {module_name} not available")(func)

    return decorator


def parametrize_with_ids(param_name: str, values: List[Any], ids: Optional[List[str]] = None):
    """
    Parametrize decorator with automatic ID generation.

    Example:
        >>> @parametrize_with_ids('n', [1, 10, 100], ids=['small', 'medium', 'large'])
        ... def test_sum(n):
        ...     assert sum(range(n)) >= 0
    """
    if ids is None:
        ids = [str(v) for v in values]
    return pytest.mark.parametrize(param_name, values, ids=ids)
