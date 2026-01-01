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

import functools
import sysconfig
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

# Cache expensive sysconfig calls
_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"


class ExtModule(BaseModel):
    """Config for a single Zig extension module."""

    name: str
    root: Path
    limited_api: bool = True

    # C/C++ integration support
    c_sources: list[str] = Field(default_factory=list, description="C/C++ source files to compile")
    c_include_dirs: list[str] = Field(default_factory=list, description="Include directories for C headers")
    c_libraries: list[str] = Field(default_factory=list, description="System libraries to link (-l)")
    c_flags: list[str] = Field(default_factory=list, description="C/C++ compiler flags")
    ld_flags: list[str] = Field(default_factory=list, description="Linker flags")
    link_all_deps: bool = Field(default=False, description="Auto-link all pyz3_deps.json dependencies")

    @property
    def libname(self) -> str:
        return self.name.rsplit(".", maxsplit=1)[-1]

    @property
    def install_path(self) -> Path:
        if self.limited_api:
            suffix = ".abi3.so"
        else:
            # Use cached platform-specific suffix (e.g., .cpython-314-darwin.so)
            suffix = _EXT_SUFFIX
        return Path(*self.name.split(".")).with_suffix(suffix)

    @property
    def test_bin(self) -> Path:
        return (Path("zig-out") / "bin" / self.libname).with_suffix(".test.bin")


class ToolPydust(BaseModel):
    """Model for tool.pyz3 section of a pyproject.toml."""

    zig_exe: Path | None = None
    build_zig: Path = Path("build.zig")

    # Whether to include Zig tests as part of the pytest collection.
    zig_tests: bool = True

    # When true, python module definitions are configured by the user in their own build.zig file.
    # When false, ext_modules is used to auto-generated a build.zig file.
    self_managed: bool = False

    # We rename pluralized config sections so the pyproject.toml reads better.
    ext_modules: list[ExtModule] = Field(alias="ext_module", default_factory=list)

    @property
    def pyz3_build_zig(self) -> Path:
        return self.build_zig.parent / "pyz3.build.zig"

    @model_validator(mode="after")
    def validate_atts(self):
        if self.self_managed and self.ext_modules:
            raise ValueError("ext_modules cannot be defined when using pyz3 in self-managed mode.")
        return self


@functools.cache
def load() -> ToolPydust:
    """Load pyz3 configuration from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found in current directory")

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    tool_config = pyproject.get("tool", {}).get("pyz3", {})
    return ToolPydust(**tool_config)
