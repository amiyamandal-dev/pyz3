import sys

from pyz3 import buildzig


def build() -> None:
    """The main entry point from Poetry's build script."""
    buildzig.zig_build(["install", f"-Dpython-exe={sys.executable}", "-Doptimize=ReleaseSafe"])
