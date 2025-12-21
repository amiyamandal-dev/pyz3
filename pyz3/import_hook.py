"""
Import hook for pyz3 - enables direct importing of .zig files from Python.

Provides IPython/Jupyter notebook support for zigimport.
"""

from typing import Any

from . import zigimport

__all__ = ["load_ipython_extension", "unload_ipython_extension"]


def load_ipython_extension(ipython: Any) -> None:
    """
    Load the zigimport extension in IPython/Jupyter.

    Usage in Jupyter:
        %load_ext pyz3.import_hook

    Then you can import .zig files directly:
        import my_module
    """
    zigimport.install()
    print("✓ zigimport enabled!")
    print("\nYou can now import .zig files directly:")
    print("  import my_module  # Compiles my_module.zig automatically\n")
    print("Environment variables:")
    print("  ZIGIMPORT_OPTIMIZE=Debug|ReleaseSafe|ReleaseFast|ReleaseSmall")
    print("  ZIGIMPORT_VERBOSE=1              # Enable verbose output")
    print("  ZIGIMPORT_FORCE_REBUILD=1        # Force rebuild on import")
    print(f"  ZIGIMPORT_BUILD_DIR={zigimport._config.build_dir}")


def unload_ipython_extension(ipython: Any) -> None:
    """Unload the zigimport extension in IPython/Jupyter."""
    zigimport.uninstall()
    print("✓ zigimport disabled")
