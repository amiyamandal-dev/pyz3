"""
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
import pytest
import importlib
from pathlib import Path

from pyz3 import buildzig
from test.helpers import PerformanceTester


@pytest.fixture
def example():
    """Fixture that provides the example module for tests."""
    import example
    # Import submodules to make them accessible as attributes
    import example.list_conversion_example
    import example.opaque_state
    return example


@pytest.fixture
def hello_module():
    """Fixture providing the hello example module."""
    from example import hello
    return hello


@pytest.fixture
def functions_module():
    """Fixture providing the functions example module."""
    from example import functions
    return functions


@pytest.fixture
def memory_module():
    """Fixture providing the memory example module."""
    from example import memory
    return memory


@pytest.fixture
def exceptions_module():
    """Fixture providing the exceptions example module."""
    from example import exceptions
    return exceptions


@pytest.fixture(autouse=True)
def cleanup_gc():
    """Automatically run garbage collection after each test."""
    yield
    gc.collect()


@pytest.fixture
def perf():
    """Fixture providing PerformanceTester instance."""
    return PerformanceTester()


@pytest.fixture
def project_root():
    """Fixture providing the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def perf_baseline(tmp_path):
    """
    Fixture for performance baseline tracking.

    Stores and loads performance baselines from a JSON file.
    """
    import json

    baseline_file = tmp_path / "perf_baseline.json"

    class BaselineTracker:
        def __init__(self, file_path):
            self.file_path = file_path
            self.baselines = {}
            if file_path.exists():
                with open(file_path) as f:
                    self.baselines = json.load(f)

        def get(self, key, default=None):
            return self.baselines.get(key, default)

        def set(self, key, value):
            self.baselines[key] = value

        def save(self):
            with open(self.file_path, 'w') as f:
                json.dump(self.baselines, f, indent=2)

        def check_regression(self, key, current_value, tolerance=1.2):
            """Check if current value is within tolerance of baseline."""
            baseline = self.get(key)
            if baseline is None:
                self.set(key, current_value)
                return True
            return current_value <= baseline * tolerance

    tracker = BaselineTracker(baseline_file)
    yield tracker
    tracker.save()


def pytest_collection(session):
    """We use the same pyz3 build system for our example modules, but we trigger it from a pytest hook."""
    # We can't use a temp-file since zig build's caching depends on the file path.
    buildzig.zig_build(["install"])


def pytest_collection_modifyitems(session, config, items):
    """The pyz3 Pytest plugin runs Zig tests from within the examples project.

    To ensure our plugin captures the failures, we have made one of those tests fail.
    Therefore we mark it here is "xfail" to test that it actually does so.
    """
    for item in items:
        if item.nodeid == "example/pytest.zig::pytest.test.pyz3-expected-failure":
            item.add_marker(pytest.mark.xfail(strict=True))
