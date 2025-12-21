import functools
import importlib.metadata
from pathlib import Path

import tomllib
from pydantic import BaseModel, Field, model_validator


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
        # Note: Non-limited API support would require platform/version-specific suffixes
        # (e.g., .cpython-312-darwin.so). For now, we only support limited API (.abi3.so)
        # which is forward-compatible across Python 3 versions.
        assert self.limited_api, "Only limited API modules are supported right now"
        return Path(*self.name.split(".")).with_suffix(".abi3.so")

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
            raise ValueError("ext_modules cannot be defined when using Pydust in self-managed mode.")
        return self


@functools.cache
def load() -> ToolPydust:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    # Since Poetry doesn't support locking the build-system.requires dependencies,
    # we perform a check here to prevent the versions from diverging.
    pyz3_version = importlib.metadata.version("pyZ3")

    # Skip development version when installed locally (e.g., pip install -e .)
    # Development installs may report version from __init__.py or 0.0.0
    if pyz3_version not in ("0.1.0", "0.8.0", "0.0.0") and not pyz3_version.endswith(".dev"):
        for req in pyproject["build-system"]["requires"]:
            if not req.startswith("pyZ3"):
                continue
            expected = f"pyZ3=={pyz3_version}"
            if req != expected:
                raise ValueError(
                    "Detected misconfigured pyZ3. "
                    f'You must include "{expected}" in build-system.requires in pyproject.toml'
                )

    return ToolPydust(**pyproject["tool"].get("pyz3", {}))
