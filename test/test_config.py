"""
Unit tests for pyz3.config module.
Tests the ExtModule and ToolPydust configuration models.
"""

import tempfile
from pathlib import Path

import pytest
import tomllib
from pydantic import ValidationError

from pyz3.config import ExtModule, ToolPydust


class TestExtModule:
    """Tests for ExtModule configuration model."""

    def test_basic_ext_module(self):
        """Test basic ExtModule creation."""
        ext_mod = ExtModule(name="mypackage.mymodule", root=Path("src/mymodule.zig"))

        assert ext_mod.name == "mypackage.mymodule"
        assert ext_mod.root == Path("src/mymodule.zig")
        assert ext_mod.limited_api is True
        assert ext_mod.c_sources == []
        assert ext_mod.c_include_dirs == []
        assert ext_mod.c_libraries == []
        assert ext_mod.c_flags == []
        assert ext_mod.ld_flags == []
        assert ext_mod.link_all_deps is False

    def test_ext_module_with_c_sources(self):
        """Test ExtModule with C sources."""
        ext_mod = ExtModule(
            name="mypackage.native",
            root=Path("src/native.zig"),
            c_sources=["src/helper.c", "src/utils.c"],
            c_include_dirs=["include/", "deps/"],
            c_libraries=["m", "pthread"],
            c_flags=["-O3", "-march=native"],
            ld_flags=["-L/usr/local/lib"],
        )

        assert ext_mod.c_sources == ["src/helper.c", "src/utils.c"]
        assert ext_mod.c_include_dirs == ["include/", "deps/"]
        assert ext_mod.c_libraries == ["m", "pthread"]
        assert ext_mod.c_flags == ["-O3", "-march=native"]
        assert ext_mod.ld_flags == ["-L/usr/local/lib"]

    def test_ext_module_link_all_deps(self):
        """Test ExtModule with link_all_deps enabled."""
        ext_mod = ExtModule(
            name="mypackage.csv",
            root=Path("src/csv.zig"),
            link_all_deps=True,
        )

        assert ext_mod.link_all_deps is True

    def test_ext_module_libname(self):
        """Test libname property."""
        ext_mod = ExtModule(name="mypackage.subpkg.mymodule", root=Path("src/mod.zig"))
        assert ext_mod.libname == "mymodule"

        ext_mod = ExtModule(name="simple", root=Path("src/simple.zig"))
        assert ext_mod.libname == "simple"

    def test_ext_module_install_path(self):
        """Test install_path property."""
        ext_mod = ExtModule(name="mypackage.mymodule", root=Path("src/mod.zig"))
        assert ext_mod.install_path == Path("mypackage/mymodule.abi3.so")

        ext_mod = ExtModule(name="pkg.sub.mod", root=Path("src/mod.zig"))
        assert ext_mod.install_path == Path("pkg/sub/mod.abi3.so")

    def test_ext_module_test_bin(self):
        """Test test_bin property."""
        ext_mod = ExtModule(name="mypackage.mymodule", root=Path("src/mod.zig"))
        assert ext_mod.test_bin == Path("zig-out/bin/mymodule.test.bin")


class TestToolPydust:
    """Tests for ToolPydust configuration model."""

    def test_basic_tool_pydust(self):
        """Test basic ToolPydust creation."""
        tool = ToolPydust(ext_module=[ExtModule(name="pkg.mod", root=Path("src/mod.zig"))])

        assert len(tool.ext_modules) == 1
        assert tool.zig_exe is None
        assert tool.build_zig == Path("build.zig")
        assert tool.zig_tests is True
        assert tool.self_managed is False

    def test_tool_pydust_custom_build_zig(self):
        """Test ToolPydust with custom build.zig path."""
        tool = ToolPydust(build_zig=Path("custom.build.zig"))

        assert tool.build_zig == Path("custom.build.zig")
        assert tool.pyz3_build_zig == Path("pyz3.build.zig")

    def test_tool_pydust_self_managed(self):
        """Test ToolPydust in self-managed mode."""
        tool = ToolPydust(self_managed=True)

        assert tool.self_managed is True
        assert len(tool.ext_modules) == 0

    def test_tool_pydust_validation_error(self):
        """Test that self_managed and ext_modules cannot both be set."""
        with pytest.raises(ValidationError, match="ext_modules cannot be defined"):
            ToolPydust(self_managed=True, ext_module=[ExtModule(name="pkg.mod", root=Path("src/mod.zig"))])


class TestConfigParsing:
    """Tests for parsing pyproject.toml configurations."""

    def test_parse_toml_with_c_config(self):
        """Test parsing TOML with C/C++ configuration."""
        toml_content = """
[tool.pyz3]
build_zig = "build.zig"

[[tool.pyz3.ext_module]]
name = "mypackage.core"
root = "src/core.zig"
c_sources = ["src/helper.c"]
c_include_dirs = ["include/"]
c_libraries = ["m"]
c_flags = ["-O2"]
ld_flags = ["-L/usr/local/lib"]
link_all_deps = true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            with open(f.name, "rb") as toml_file:
                data = tomllib.load(toml_file)

            tool = ToolPydust(**data["tool"]["pyz3"])

            assert len(tool.ext_modules) == 1
            ext_mod = tool.ext_modules[0]
            assert ext_mod.name == "mypackage.core"
            assert ext_mod.c_sources == ["src/helper.c"]
            assert ext_mod.c_include_dirs == ["include/"]
            assert ext_mod.c_libraries == ["m"]
            assert ext_mod.c_flags == ["-O2"]
            assert ext_mod.ld_flags == ["-L/usr/local/lib"]
            assert ext_mod.link_all_deps is True

            # Cleanup
            Path(f.name).unlink()

    def test_parse_toml_without_c_config(self):
        """Test parsing TOML without C/C++ configuration (defaults)."""
        toml_content = """
[tool.pyz3]

[[tool.pyz3.ext_module]]
name = "mypackage.simple"
root = "src/simple.zig"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            with open(f.name, "rb") as toml_file:
                data = tomllib.load(toml_file)

            tool = ToolPydust(**data["tool"]["pyz3"])

            assert len(tool.ext_modules) == 1
            ext_mod = tool.ext_modules[0]
            assert ext_mod.name == "mypackage.simple"
            assert ext_mod.c_sources == []
            assert ext_mod.c_include_dirs == []
            assert ext_mod.c_libraries == []
            assert ext_mod.c_flags == []
            assert ext_mod.ld_flags == []
            assert ext_mod.link_all_deps is False

            # Cleanup
            Path(f.name).unlink()

    def test_parse_toml_multiple_modules_mixed(self):
        """Test parsing TOML with multiple modules, some with C config."""
        toml_content = """
[tool.pyz3]

[[tool.pyz3.ext_module]]
name = "mypackage.pure_zig"
root = "src/pure.zig"

[[tool.pyz3.ext_module]]
name = "mypackage.with_c"
root = "src/with_c.zig"
c_sources = ["src/helper.c"]
c_libraries = ["sqlite3"]

[[tool.pyz3.ext_module]]
name = "mypackage.another_zig"
root = "src/another.zig"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            with open(f.name, "rb") as toml_file:
                data = tomllib.load(toml_file)

            tool = ToolPydust(**data["tool"]["pyz3"])

            assert len(tool.ext_modules) == 3

            # First module - pure Zig
            assert tool.ext_modules[0].name == "mypackage.pure_zig"
            assert tool.ext_modules[0].c_sources == []

            # Second module - with C
            assert tool.ext_modules[1].name == "mypackage.with_c"
            assert tool.ext_modules[1].c_sources == ["src/helper.c"]
            assert tool.ext_modules[1].c_libraries == ["sqlite3"]

            # Third module - pure Zig
            assert tool.ext_modules[2].name == "mypackage.another_zig"
            assert tool.ext_modules[2].c_sources == []

            # Cleanup
            Path(f.name).unlink()
