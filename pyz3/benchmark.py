"""Benchmarking utilities for pyz3 extension modules.

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

import importlib
import inspect
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyz3.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single function benchmark."""

    function_name: str
    iterations: int
    total_time: float
    mean_time: float
    std_dev: float
    min_time: float
    max_time: float
    median_time: float
    throughput: float  # calls per second

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "function": self.function_name,
            "iterations": self.iterations,
            "total_time_s": self.total_time,
            "mean_time_us": self.mean_time * 1e6,
            "std_dev_us": self.std_dev * 1e6,
            "min_time_us": self.min_time * 1e6,
            "max_time_us": self.max_time * 1e6,
            "median_time_us": self.median_time * 1e6,
            "throughput_calls_per_sec": self.throughput,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    module_name: str
    results: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "module": self.module_name,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def _get_test_args(func: Callable) -> dict[str, Any]:
    """Generate test arguments for a function based on its signature."""
    sig = inspect.signature(func)
    args = {}

    for name, param in sig.parameters.items():
        if param.annotation is inspect.Parameter.empty:
            # No annotation, try common defaults
            args[name] = 0
        elif param.annotation is int:
            args[name] = 42
        elif param.annotation is float:
            args[name] = 3.14
        elif param.annotation is str:
            args[name] = "test"
        elif param.annotation is bool:
            args[name] = True
        elif param.annotation is list:
            args[name] = [1, 2, 3]
        elif param.annotation is dict:
            args[name] = {"key": "value"}
        else:
            # Default to None or 0 for unknown types
            args[name] = 0

    return args


def benchmark_function(
    func: Callable,
    iterations: int = 10000,
    warmup: int = 100,
    args: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """
    Benchmark a single function.

    Args:
        func: Function to benchmark
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
        args: Arguments to pass to the function (auto-generated if None)

    Returns:
        BenchmarkResult with timing statistics
    """
    if args is None:
        args = _get_test_args(func)

    # Warmup
    for _ in range(warmup):
        try:
            func(**args)
        except Exception:
            # If default args don't work, try with no args
            try:
                func()
                args = {}
            except Exception:
                raise ValueError(f"Cannot auto-benchmark {func.__name__}: please provide explicit arguments")

    # Collect timing samples
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(**args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    total_time = sum(times)
    mean_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
    min_time = min(times)
    max_time = max(times)
    median_time = statistics.median(times)
    throughput = iterations / total_time if total_time > 0 else 0

    return BenchmarkResult(
        function_name=func.__name__,
        iterations=iterations,
        total_time=total_time,
        mean_time=mean_time,
        std_dev=std_dev,
        min_time=min_time,
        max_time=max_time,
        median_time=median_time,
        throughput=throughput,
    )


def benchmark_module(
    module_name: str,
    function_name: str | None = None,
    iterations: int = 10000,
    warmup: int = 100,
) -> BenchmarkSuite:
    """
    Benchmark functions in a module.

    Args:
        module_name: Name of the module to benchmark
        function_name: Specific function to benchmark (None = all public functions)
        iterations: Number of iterations per function
        warmup: Number of warmup iterations

    Returns:
        BenchmarkSuite with all results
    """
    logger.info(f"Loading module: {module_name}")
    module = importlib.import_module(module_name)

    suite = BenchmarkSuite(module_name=module_name)

    # Get functions to benchmark
    if function_name:
        if not hasattr(module, function_name):
            raise ValueError(f"Function '{function_name}' not found in {module_name}")
        functions = [(function_name, getattr(module, function_name))]
    else:
        # Get all public callable attributes
        functions = [
            (name, obj) for name, obj in inspect.getmembers(module) if callable(obj) and not name.startswith("_")
        ]

    for name, func in functions:
        logger.info(f"Benchmarking: {name}")
        try:
            result = benchmark_function(func, iterations=iterations, warmup=warmup)
            suite.results.append(result)
        except Exception as e:
            logger.warning(f"Skipping {name}: {e}")

    return suite


def print_results(suite: BenchmarkSuite) -> None:
    """Print benchmark results in a formatted table."""
    print(f"\n{'=' * 70}")
    print(f"Benchmark Results: {suite.module_name}")
    print(f"Timestamp: {suite.timestamp}")
    print(f"{'=' * 70}\n")

    if not suite.results:
        print("No functions benchmarked.")
        return

    # Header
    print(f"{'Function':<30} {'Mean (µs)':>12} {'Std Dev':>12} {'Throughput':>15}")
    print(f"{'-' * 30} {'-' * 12} {'-' * 12} {'-' * 15}")

    for r in suite.results:
        mean_us = r.mean_time * 1e6
        std_us = r.std_dev * 1e6
        throughput = f"{r.throughput:,.0f}/s"
        print(f"{r.function_name:<30} {mean_us:>12.3f} {std_us:>12.3f} {throughput:>15}")

    print(f"\n{'=' * 70}")
    print(f"Total functions benchmarked: {len(suite.results)}")
    print(f"Iterations per function: {suite.results[0].iterations if suite.results else 0}")
    print(f"{'=' * 70}\n")


def run_benchmark(
    module_name: str,
    function_name: str | None = None,
    iterations: int = 10000,
    warmup: int = 100,
    output_json: bool = False,
    output_file: str | None = None,
) -> BenchmarkSuite:
    """
    Run benchmarks and display/save results.

    Args:
        module_name: Module to benchmark
        function_name: Specific function (optional)
        iterations: Number of iterations
        warmup: Warmup iterations
        output_json: Output as JSON
        output_file: Save results to file

    Returns:
        BenchmarkSuite with results
    """
    suite = benchmark_module(
        module_name=module_name,
        function_name=function_name,
        iterations=iterations,
        warmup=warmup,
    )

    if output_json:
        output = suite.to_json()
        print(output)
    else:
        print_results(suite)

    if output_file:
        path = Path(output_file)
        path.write_text(suite.to_json())
        logger.info(f"Results saved to: {path}")

    return suite
