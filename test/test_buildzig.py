"""
Unit tests for pyz3.buildzig module.
Tests build.zig generation with C/C++ configuration.
"""

import io
from pathlib import Path

from pyz3.buildzig import _format_zig_array, generate_build_zig
from pyz3.config import ExtModule, ToolPydust


class TestFormatZigArray:
    """Tests for _format_zig_array helper function."""

    def test_empty_array(self):
        """Test formatting empty array."""
        result = _format_zig_array([])
        assert result == "&.{}"

    def test_single_item(self):
        """Test formatting single item array."""
        result = _format_zig_array(["item"])
        assert result == '&.{ "item" }'

    def test_multiple_items(self):
        """Test formatting multiple items array."""
        result = _format_zig_array(["item1", "item2", "item3"])
        assert result == '&.{ "item1", "item2", "item3" }'

    def test_items_with_paths(self):
        """Test formatting array with file paths."""
        result = _format_zig_array(["src/file1.c", "src/file2.c"])
        assert result == '&.{ "src/file1.c", "src/file2.c" }'

    def test_items_with_flags(self):
        """Test formatting array with compiler flags."""
        result = _format_zig_array(["-O3", "-march=native", "-DNDEBUG"])
        assert result == '&.{ "-O3", "-march=native", "-DNDEBUG" }'


class TestGenerateBuildZig:
    """Tests for generate_build_zig function."""

    def test_generate_simple_module(self):
        """Test generating build.zig for simple module without C config."""
        conf = ToolPydust(ext_module=[ExtModule(name="example.hello", root=Path("example/hello.zig"))])

        output = io.StringIO()
        generate_build_zig(output, conf)
        result = output.getvalue()

        # Check basic structure
        assert 'const std = @import("std");' in result
        assert 'const py = @import("./pyz3.build.zig");' in result
        assert "pub fn build(b: *std.Build) void" in result
        assert "pyz3.addPythonModule" in result

        # Check module configuration
        assert '.name = "example.hello"' in result
        assert '.root_source_file = b.path("example/hello.zig")' in result
        assert ".limited_api = true" in result

        # Check C/C++ arrays are empty
        assert ".c_sources = &.{}" in result
        assert ".c_include_dirs = &.{}" in result
        assert ".c_libraries = &.{}" in result
        assert ".c_flags = &.{}" in result
        assert ".ld_flags = &.{}" in result

    def test_generate_module_with_c_sources(self):
        """Test generating build.zig for module with C sources."""
        conf = ToolPydust(
            ext_module=[
                ExtModule(
                    name="example.c_integration",
                    root=Path("example/c_integration.zig"),
                    c_sources=["example/helper.c", "example/utils.c"],
                    c_include_dirs=["example/", "include/"],
                    c_libraries=["m", "pthread"],
                    c_flags=["-O3", "-march=native"],
                    ld_flags=["-L/usr/local/lib"],
                )
            ]
        )

        output = io.StringIO()
        generate_build_zig(output, conf)
        result = output.getvalue()

        # Check C sources
        assert '.c_sources = &.{ "example/helper.c", "example/utils.c" }' in result

        # Check include directories
        assert '.c_include_dirs = &.{ "example/", "include/" }' in result

        # Check libraries
        assert '.c_libraries = &.{ "m", "pthread" }' in result

        # Check C flags
        assert '.c_flags = &.{ "-O3", "-march=native" }' in result

        # Check linker flags
        assert '.ld_flags = &.{ "-L/usr/local/lib" }' in result

    def test_generate_multiple_modules_mixed(self):
        """Test generating build.zig for multiple modules with mixed config."""
        conf = ToolPydust(
            ext_module=[
                ExtModule(name="example.pure_zig", root=Path("example/pure.zig")),
                ExtModule(
                    name="example.with_c",
                    root=Path("example/with_c.zig"),
                    c_sources=["example/helper.c"],
                    c_libraries=["sqlite3"],
                ),
                ExtModule(name="example.another", root=Path("example/another.zig")),
            ]
        )

        output = io.StringIO()
        generate_build_zig(output, conf)
        result = output.getvalue()

        # Check all three modules are present
        assert '.name = "example.pure_zig"' in result
        assert '.name = "example.with_c"' in result
        assert '.name = "example.another"' in result

        # Count addPythonModule calls
        assert result.count("pyz3.addPythonModule") == 3

    def test_generate_windows_path_conversion(self):
        """Test that Windows backslashes are converted to forward slashes."""
        conf = ToolPydust(ext_module=[ExtModule(name="example.test", root=Path("example\\test.zig"))])

        output = io.StringIO()
        generate_build_zig(output, conf)
        result = output.getvalue()

        # Check path uses forward slashes
        assert '.root_source_file = b.path("example/test.zig")' in result
        assert "\\" not in result  # No backslashes in output

    def test_generate_module_with_only_libraries(self):
        """Test module that only links system libraries."""
        conf = ToolPydust(
            ext_module=[
                ExtModule(
                    name="example.sqlite",
                    root=Path("example/sqlite.zig"),
                    c_libraries=["sqlite3"],
                )
            ]
        )

        output = io.StringIO()
        generate_build_zig(output, conf)
        result = output.getvalue()

        # Check only libraries are set, others are empty
        assert ".c_sources = &.{}" in result
        assert ".c_include_dirs = &.{}" in result
        assert '.c_libraries = &.{ "sqlite3" }' in result
        assert ".c_flags = &.{}" in result
        assert ".ld_flags = &.{}" in result

    def test_generate_module_with_only_include_dirs(self):
        """Test module that only adds include directories."""
        conf = ToolPydust(
            ext_module=[
                ExtModule(
                    name="example.headers_only",
                    root=Path("example/headers.zig"),
                    c_include_dirs=["deps/mylib/include"],
                )
            ]
        )

        output = io.StringIO()
        generate_build_zig(output, conf)
        result = output.getvalue()

        # Check only include_dirs are set
        assert ".c_sources = &.{}" in result
        assert '.c_include_dirs = &.{ "deps/mylib/include" }' in result
        assert ".c_libraries = &.{}" in result

    def test_generate_preserves_formatting(self):
        """Test that generated build.zig is properly formatted."""
        conf = ToolPydust(
            ext_module=[
                ExtModule(
                    name="example.test",
                    root=Path("example/test.zig"),
                    c_sources=["src/test.c"],
                )
            ]
        )

        output = io.StringIO()
        generate_build_zig(output, conf)
        result = output.getvalue()

        # Check proper indentation exists
        assert "    " in result  # Should have indented content
        # Check no trailing whitespace on empty lines
        lines = result.split("\n")
        for line in lines:
            if not line.strip():
                assert line == "", f"Empty line has trailing whitespace: {repr(line)}"
