"""Memory leak detection utilities for pyz3 extension modules.

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

import gc
import os
import subprocess
import sys
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

from pyz3.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Represents a memory snapshot at a point in time."""

    current: int  # Current memory usage in bytes
    peak: int  # Peak memory usage in bytes
    traced_objects: int  # Number of traced memory blocks


@dataclass
class LeakReport:
    """Report of potential memory leaks."""

    script: str
    initial_memory: MemorySnapshot
    final_memory: MemorySnapshot
    growth: int  # Memory growth in bytes
    top_allocations: list[tuple[str, int]]  # (traceback, size) pairs
    potential_leaks: list[tuple[str, int]]  # Allocations that grew significantly

    def is_leak_suspected(self, threshold: int = 1024) -> bool:
        """Check if memory growth exceeds threshold."""
        return self.growth > threshold

    def print_report(self) -> None:
        """Print a formatted leak report."""
        print("\n" + "=" * 70)
        print("Memory Leak Detection Report")
        print("=" * 70)
        print(f"\nScript: {self.script}")
        print(f"\nInitial Memory: {self.initial_memory.current / 1024:.2f} KB")
        print(f"Final Memory:   {self.final_memory.current / 1024:.2f} KB")
        print(f"Peak Memory:    {self.final_memory.peak / 1024:.2f} KB")
        print(f"Memory Growth:  {self.growth / 1024:.2f} KB")

        if self.potential_leaks:
            print(f"\n{'!' * 70}")
            print("POTENTIAL MEMORY LEAKS DETECTED")
            print(f"{'!' * 70}")
            for tb, size in self.potential_leaks[:10]:
                print(f"\n[{size / 1024:.2f} KB]")
                print(tb)
        else:
            print("\n[OK] No significant memory leaks detected.")

        if self.top_allocations:
            print("\n" + "-" * 70)
            print("Top Memory Allocations:")
            print("-" * 70)
            for tb, size in self.top_allocations[:5]:
                print(f"\n[{size / 1024:.2f} KB]")
                # Only show first few lines of traceback
                lines = tb.split("\n")[:4]
                print("\n".join(lines))

        print("\n" + "=" * 70)


def take_snapshot() -> MemorySnapshot:
    """Take a memory snapshot."""
    current, peak = tracemalloc.get_traced_memory()
    stats = tracemalloc.get_traceback_memory()
    return MemorySnapshot(
        current=current,
        peak=peak,
        traced_objects=len(stats) if stats else 0,
    )


def run_with_memcheck(
    script_path: str,
    threshold: int = 1024,
    verbose: bool = False,
) -> LeakReport:
    """
    Run a Python script with memory leak detection.

    Args:
        script_path: Path to the Python script
        threshold: Minimum leak size to report (bytes)
        verbose: Enable verbose output

    Returns:
        LeakReport with memory analysis
    """
    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Start tracing
    tracemalloc.start(25)  # Keep 25 frames of traceback

    # Force garbage collection before starting
    gc.collect()

    initial = take_snapshot()

    if verbose:
        logger.info(f"Running script: {script}")
        logger.info(f"Initial memory: {initial.current / 1024:.2f} KB")

    # Execute the script
    script_globals = {
        "__name__": "__main__",
        "__file__": str(script.absolute()),
    }

    try:
        exec(compile(script.read_text(), script, "exec"), script_globals)
    except Exception as e:
        logger.error(f"Script error: {e}")
        raise

    # Force garbage collection after script
    gc.collect()
    gc.collect()
    gc.collect()

    final = take_snapshot()

    if verbose:
        logger.info(f"Final memory: {final.current / 1024:.2f} KB")

    # Get top memory allocations
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("traceback")

    top_allocations = []
    for stat in top_stats[:20]:
        tb = "\n".join(stat.traceback.format())
        top_allocations.append((tb, stat.size))

    # Find potential leaks (allocations that are significant)
    potential_leaks = [
        (tb, size) for tb, size in top_allocations if size > threshold
    ]

    tracemalloc.stop()

    return LeakReport(
        script=str(script),
        initial_memory=initial,
        final_memory=final,
        growth=final.current - initial.current,
        top_allocations=top_allocations,
        potential_leaks=potential_leaks,
    )


def run_subprocess_memcheck(
    script_path: str,
    threshold: int = 1024,
    verbose: bool = False,
) -> int:
    """
    Run memory check in a subprocess for isolation.

    Args:
        script_path: Path to the Python script
        threshold: Minimum leak size to report
        verbose: Enable verbose output

    Returns:
        Exit code (0 = no leaks, 1 = leaks detected)
    """
    # Create a wrapper script that runs memcheck
    wrapper_code = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent}")
from pyz3.memcheck import run_with_memcheck

report = run_with_memcheck(
    "{script_path}",
    threshold={threshold},
    verbose={verbose},
)
report.print_report()
sys.exit(1 if report.is_leak_suspected({threshold}) else 0)
'''

    env = os.environ.copy()
    env["PYTHONMALLOC"] = "malloc"  # Use system malloc for better tracking

    result = subprocess.run(
        [sys.executable, "-c", wrapper_code],
        env=env,
        capture_output=not verbose,
    )

    return result.returncode


def memcheck_module(
    module_name: str,
    iterations: int = 100,
    threshold: int = 1024,
    verbose: bool = False,
) -> LeakReport:
    """
    Check a module for memory leaks by repeatedly calling its functions.

    Args:
        module_name: Name of the module to check
        iterations: Number of iterations per function
        threshold: Minimum leak size to report
        verbose: Enable verbose output

    Returns:
        LeakReport with results
    """
    import importlib
    import inspect

    tracemalloc.start(25)
    gc.collect()

    initial = take_snapshot()

    # Import and exercise the module
    module = importlib.import_module(module_name)

    # Get all public callables
    functions = [
        (name, obj)
        for name, obj in inspect.getmembers(module)
        if callable(obj) and not name.startswith("_")
    ]

    for name, func in functions:
        if verbose:
            logger.info(f"Testing: {name}")

        for _ in range(iterations):
            try:
                # Try calling with no args first
                func()
            except TypeError:
                # Needs args, skip for now
                break
            except Exception:
                break

        gc.collect()

    gc.collect()
    gc.collect()

    final = take_snapshot()

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("traceback")

    top_allocations = [(
        "\n".join(stat.traceback.format()),
        stat.size,
    ) for stat in top_stats[:20]]

    potential_leaks = [
        (tb, size) for tb, size in top_allocations if size > threshold
    ]

    tracemalloc.stop()

    return LeakReport(
        script=f"module:{module_name}",
        initial_memory=initial,
        final_memory=final,
        growth=final.current - initial.current,
        top_allocations=top_allocations,
        potential_leaks=potential_leaks,
    )


def get_traceback_memory() -> dict:
    """Get memory statistics by traceback.

    This is a helper that wraps tracemalloc functionality.
    """
    if not tracemalloc.is_tracing():
        return {}

    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics("traceback")

    return {
        "\n".join(stat.traceback.format()): stat.size
        for stat in stats
    }
