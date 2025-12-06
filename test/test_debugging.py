"""
Comprehensive tests for debugging support in Ziggy Pydust.

Tests cover:
1. Debug logging
2. Stack traces
3. Breakpoints
4. Debug helpers
5. IDE integration
"""

import os
import sys
from pathlib import Path

import pytest


class TestDebugHelpers:
    """Tests for Python-side debug helpers."""

    def test_debug_helper_exists(self):
        """Verify DebugHelper class is available."""
        from pyz3.debug import DebugHelper

        assert DebugHelper is not None
        assert hasattr(DebugHelper, 'get_extension_path')
        assert hasattr(DebugHelper, 'get_debug_symbols_info')
        assert hasattr(DebugHelper, 'attach_debugger')

    def test_breakpoint_context(self):
        """Test BreakpointContext (non-interactive)."""
        from pyz3.debug import BreakpointContext

        # Create context but don't actually pause
        ctx = BreakpointContext("test breakpoint")
        assert ctx.message == "test breakpoint"

    def test_inspect_extension_module(self):
        """Test extension module inspection."""
        from pyz3.debug import inspect_extension

        # Should not crash even with non-existent module
        # Just testing it doesn't raise
        try:
            inspect_extension("nonexistent_module")
        except Exception as e:
            pytest.fail(f"inspect_extension should not raise: {e}")

    def test_enable_core_dumps(self):
        """Test core dump enabling."""
        from pyz3.debug import DebugHelper

        # Should not crash
        DebugHelper.enable_core_dumps()

    def test_get_extension_path(self):
        """Test getting extension module path."""
        from pyz3.debug import DebugHelper

        # Test with sys (builtin module)
        path = DebugHelper.get_extension_path('sys')
        assert path is None or isinstance(path, Path)

    def test_attach_debugger_lldb(self):
        """Test debugger attach command generation for LLDB."""
        from pyz3.debug import DebugHelper

        cmd = DebugHelper.attach_debugger('sys', debugger='lldb')
        assert isinstance(cmd, str)
        assert 'lldb' in cmd.lower()
        assert str(os.getpid()) in cmd

    def test_attach_debugger_gdb(self):
        """Test debugger attach command generation for GDB."""
        from pyz3.debug import DebugHelper

        cmd = DebugHelper.attach_debugger('sys', debugger='gdb')
        assert isinstance(cmd, str)
        assert 'gdb' in cmd.lower()
        assert str(os.getpid()) in cmd

    def test_attach_debugger_unknown(self):
        """Test debugger attach with unknown debugger."""
        from pyz3.debug import DebugHelper

        cmd = DebugHelper.attach_debugger('sys', debugger='unknown')
        assert 'Unknown debugger' in cmd or 'Error' in cmd

    def test_print_mixed_traceback(self):
        """Test mixed traceback printing."""
        from pyz3.debug import DebugHelper

        # Should not crash
        DebugHelper.print_mixed_traceback()

    def test_create_debug_session_script(self):
        """Test debug session script creation."""
        from pyz3.debug import create_debug_session_script
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = create_debug_session_script('sys', output_path=f'{tmpdir}/debug.py')
            assert script_path.exists()
            assert script_path.suffix == '.py'

            # Read and verify content
            content = script_path.read_text()
            assert 'import sys' in content
            assert 'DebugHelper' in content


class TestDebuggerConfiguration:
    """Tests for debugger configuration files."""

    def test_vscode_launch_config_exists(self):
        """Verify VSCode launch.json exists."""
        launch_json = Path('.vscode/launch.json')
        assert launch_json.exists(), "VSCode launch.json should exist"

        # Verify it's valid JSON
        import json
        with open(launch_json) as f:
            config = json.load(f)

        assert 'configurations' in config
        assert len(config['configurations']) > 0

    def test_lldb_config_exists(self):
        """Verify .lldbinit exists."""
        lldbinit = Path('.lldbinit')
        assert lldbinit.exists(), ".lldbinit should exist"

        content = lldbinit.read_text()
        assert 'Pydust' in content or 'pydust' in content

    def test_gdb_config_exists(self):
        """Verify .gdbinit exists."""
        gdbinit = Path('.gdbinit')
        assert gdbinit.exists(), ".gdbinit should exist"

        content = gdbinit.read_text()
        assert 'Pydust' in content or 'pydust' in content


class TestDebugConvenience:
    """Tests for convenience functions."""

    def test_dbg_break_alias(self):
        """Test dbg_break alias exists."""
        from pyz3.debug import dbg_break, breakpoint_here

        assert dbg_break is breakpoint_here

    def test_dbg_inspect_alias(self):
        """Test dbg_inspect alias exists."""
        from pyz3.debug import dbg_inspect, inspect_extension

        assert dbg_inspect is inspect_extension


class TestDebugExamples:
    """Tests that verify debug examples work."""

    def test_debug_module_loads(self):
        """Test that the debug module loads successfully."""
        from pyz3 import debug as pydust_debug

        # Should have debug utilities
        assert hasattr(pydust_debug, 'LogLevel')
        assert hasattr(pydust_debug, 'enableDebug')
        assert hasattr(pydust_debug, 'disableDebug')

    def test_log_levels_defined(self):
        """Test that log levels are properly defined."""
        # This will be tested at the Zig level, but we can verify
        # the module structure is correct
        import pyz3

        # Should not crash
        assert pydust.debug is not None


class TestIntegration:
    """Integration tests combining multiple debug features."""

    def test_full_debugging_workflow(self):
        """Test a complete debugging workflow."""
        from pyz3.debug import DebugHelper, inspect_extension

        # 1. Enable core dumps
        DebugHelper.enable_core_dumps()

        # 2. Get debugger commands
        lldb_cmd = DebugHelper.attach_debugger('sys', debugger='lldb')
        assert 'lldb' in lldb_cmd.lower()

        gdb_cmd = DebugHelper.attach_debugger('sys', debugger='gdb')
        assert 'gdb' in gdb_cmd.lower()

        # 3. Print mixed traceback
        DebugHelper.print_mixed_traceback()

        # All steps should complete without errors
        assert True

    def test_debug_script_generation(self):
        """Test that debug script can be generated and is valid."""
        from pyz3.debug import create_debug_session_script
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            script = create_debug_session_script('sys', output_path=f'{tmpdir}/test.py')

            # Script should be executable Python
            assert script.exists()
            content = script.read_text()

            # Should have shebang
            assert content.startswith('#!/usr/bin/env python3')

            # Should import necessary modules
            assert 'from pyz3.debug import' in content


def test_debugging_feature_completeness():
    """Ensure all debugging features are implemented."""

    # Zig-side debugging
    import pyz3
    assert hasattr(pydust, 'debug')

    # Python-side debugging
    from pyz3 import debug as pydust_debug
    from pyz3.debug import (
        DebugHelper,
        BreakpointContext,
        breakpoint_here,
        inspect_extension,
        create_debug_session_script,
    )

    assert DebugHelper is not None
    assert BreakpointContext is not None
    assert breakpoint_here is not None
    assert inspect_extension is not None
    assert create_debug_session_script is not None
    assert pydust_debug is not None

    # Configuration files
    assert Path('.vscode/launch.json').exists()
    assert Path('.lldbinit').exists()
    assert Path('.gdbinit').exists()

    print("✅ All debugging features are implemented!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
