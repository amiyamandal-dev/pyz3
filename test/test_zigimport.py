"""
Tests for zigimport automatic import functionality.
"""

import pytest
import sys
import os
from pathlib import Path


def test_zigimport_install_uninstall():
    """Test install/uninstall of import hook."""
    from pyz3 import zigimport

    # Should be installed automatically on import
    assert zigimport._finder is not None
    assert zigimport._finder in sys.meta_path

    # Uninstall
    zigimport.uninstall()
    assert zigimport._finder not in sys.meta_path

    # Reinstall
    zigimport.install()
    assert zigimport._finder is not None
    assert zigimport._finder in sys.meta_path


def test_zigimport_config():
    """Test configuration options."""
    from pyz3 import zigimport

    # Test default config
    config = zigimport.ZigImportConfig()
    assert config.optimize == "Debug"  # Default
    assert config.verbose is False
    assert config.force_rebuild is False
    assert config.build_dir.exists()


def test_zigimport_optimize_env():
    """Test ZIGIMPORT_OPTIMIZE environment variable."""
    # Save original
    original = os.environ.get("ZIGIMPORT_OPTIMIZE")

    try:
        # Test different optimization levels
        for level in ["Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall"]:
            os.environ["ZIGIMPORT_OPTIMIZE"] = level
            from pyz3.zigimport import ZigImportConfig
            config = ZigImportConfig()
            assert config.optimize == level

    finally:
        # Restore
        if original:
            os.environ["ZIGIMPORT_OPTIMIZE"] = original
        elif "ZIGIMPORT_OPTIMIZE" in os.environ:
            del os.environ["ZIGIMPORT_OPTIMIZE"]


def test_zigimport_verbose_env():
    """Test ZIGIMPORT_VERBOSE environment variable."""
    original = os.environ.get("ZIGIMPORT_VERBOSE")

    try:
        # Test enabled
        os.environ["ZIGIMPORT_VERBOSE"] = "1"
        from pyz3.zigimport import ZigImportConfig
        config = ZigImportConfig()
        assert config.verbose is True

        # Test disabled
        os.environ["ZIGIMPORT_VERBOSE"] = "0"
        config = ZigImportConfig()
        assert config.verbose is False

    finally:
        if original:
            os.environ["ZIGIMPORT_VERBOSE"] = original
        elif "ZIGIMPORT_VERBOSE" in os.environ:
            del os.environ["ZIGIMPORT_VERBOSE"]


def test_zigimport_cache():
    """Test cache functionality."""
    from pyz3 import zigimport

    config = zigimport.ZigImportConfig()
    cache = zigimport.ZigModuleCache(config)

    # Create a test file
    test_file = config.build_dir / "test.zig"
    test_file.write_text("test")

    # Update cache
    cache.update("test_module", test_file)

    # Check cache was saved with correct structure
    assert cache.cache.get("test_module") is not None
    assert "hash" in cache.cache["test_module"]
    assert "source" in cache.cache["test_module"]
    assert "timestamp" in cache.cache["test_module"]
    assert cache.cache["test_module"]["source"] == str(test_file)

    # Clean up
    test_file.unlink()


def test_zigimport_clear_cache():
    """Test cache clearing."""
    from pyz3 import zigimport

    # Ensure config exists
    config = zigimport.ZigImportConfig()
    config.build_dir.mkdir(parents=True, exist_ok=True)

    # Create a test file
    test_file = config.build_dir / "test.txt"
    test_file.write_text("test")

    assert test_file.exists()

    # Clear cache
    zigimport.clear_cache()

    # Build dir should still exist but be empty
    assert config.build_dir.exists()
    assert not test_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
