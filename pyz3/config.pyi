"""Type stubs for pyz3.config module."""

from pathlib import Path

class ExtModule:
    """Configuration for an extension module."""

    name: str
    root: Path
    limited_api: bool
    c_sources: list[str]
    c_include_dirs: list[str]
    c_flags: list[str]
    ld_flags: list[str]
    link_libraries: list[str]
    link_all_deps: bool
    libname: str

    @property
    def install_path(self) -> Path: ...
    @property
    def test_bin(self) -> str: ...

class ToolPydust:
    """Main configuration class for pyz3 projects."""

    root: str | None
    build_zig: Path
    pyz3_build_zig: Path
    zig_exe: str | None
    self_managed: bool
    ext_modules: list[ExtModule]

    def __init__(
        self,
        root: str | None = None,
        build_zig: str = "build.zig",
        pyz3_build_zig: str = "pyz3.build.zig",
        zig_exe: str | None = None,
        self_managed: bool = False,
        ext_modules: list[dict] | None = None,
    ) -> None: ...

def load() -> ToolPydust:
    """Load pyz3 configuration from pyproject.toml."""
    ...
