"""
Diagnostic tools for pyz3 development and troubleshooting.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import importlib.util


class Diagnostics:
    """Diagnostic utilities for pyz3 projects."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.pyproject_path = self.project_root / "pyproject.toml"

    def check_environment(self) -> Dict[str, Any]:
        """Check development environment setup."""
        checks = {}

        # Python version
        checks["python_version"] = {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "executable": sys.executable,
            "status": "✅" if sys.version_info >= (3, 11) else "⚠️  Upgrade to Python 3.11+"
        }

        # Zig installation
        try:
            result = subprocess.run(
                ["zig", "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            zig_version = result.stdout.strip() if result.returncode == 0 else "Not found"
            checks["zig"] = {
                "version": zig_version,
                "status": "✅" if result.returncode == 0 else "❌ Install Zig"
            }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            checks["zig"] = {"version": "Not found", "status": "❌ Install Zig"}

        # Poetry
        try:
            result = subprocess.run(
                ["poetry", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            poetry_version = result.stdout.strip() if result.returncode == 0 else "Not found"
            checks["poetry"] = {
                "version": poetry_version,
                "status": "✅" if result.returncode == 0 else "⚠️  Install Poetry (optional)"
            }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            checks["poetry"] = {"version": "Not found", "status": "⚠️  Install Poetry (optional)"}

        # Pyproject.toml
        checks["pyproject_toml"] = {
            "exists": self.pyproject_path.exists(),
            "status": "✅" if self.pyproject_path.exists() else "❌ Missing pyproject.toml"
        }

        return checks

    def validate_module_config(self) -> Dict[str, List[str]]:
        """Validate module configuration in pyproject.toml."""
        errors = []
        warnings = []

        if not self.pyproject_path.exists():
            errors.append("pyproject.toml not found")
            return {"errors": errors, "warnings": warnings}

        try:
            import tomllib
            with open(self.pyproject_path, "rb") as f:
                config = tomllib.load(f)

            if "tool" not in config or "pyz3" not in config["tool"]:
                warnings.append("No [tool.pyz3] section found")
                return {"errors": errors, "warnings": warnings}

            pyz3_config = config["tool"]["pyz3"]

            # Check ext_module definitions
            if "ext_module" in pyz3_config:
                for idx, module in enumerate(pyz3_config["ext_module"]):
                    # Check module name
                    if "name" not in module:
                        errors.append(f"Module #{idx}: missing 'name' field")

                    # Check root file exists
                    if "root" in module:
                        root_path = self.project_root / module["root"]
                        if not root_path.exists():
                            errors.append(f"Module '{module.get('name', f'#{idx}')}': "
                                        f"root file not found: {module['root']}")

                    # Check C sources exist
                    if "c_sources" in module:
                        for src in module["c_sources"]:
                            src_path = self.project_root / src
                            if not src_path.exists():
                                warnings.append(f"Module '{module.get('name', f'#{idx}')}': "
                                              f"C source not found: {src}")

        except Exception as e:
            errors.append(f"Error parsing pyproject.toml: {e}")

        return {"errors": errors, "warnings": warnings}

    def analyze_build_artifacts(self) -> Dict[str, Any]:
        """Analyze compiled build artifacts."""
        artifacts = {
            "modules": [],
            "total_size": 0,
            "cache_size": 0
        }

        # Check example directory for .so files
        example_dir = self.project_root / "example"
        if example_dir.exists():
            for so_file in example_dir.glob("*.so"):
                size = so_file.stat().st_size
                artifacts["modules"].append({
                    "name": so_file.stem.replace(".abi3", ""),
                    "path": str(so_file.relative_to(self.project_root)),
                    "size": size,
                    "size_mb": round(size / (1024 * 1024), 2)
                })
                artifacts["total_size"] += size

        # Check cache size
        cache_dir = self.project_root / ".zig-cache"
        if cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            artifacts["cache_size"] = cache_size
            artifacts["cache_size_mb"] = round(cache_size / (1024 * 1024), 2)

        artifacts["total_size_mb"] = round(artifacts["total_size"] / (1024 * 1024), 2)

        return artifacts

    def check_type_coverage(self) -> Dict[str, Any]:
        """Check which Python types are supported by pyz3."""
        pyz3_src = self.project_root / "pyz3" / "src" / "types"

        if not pyz3_src.exists():
            return {"error": "pyz3/src/types directory not found"}

        supported_types = set()
        for zig_file in pyz3_src.glob("*.zig"):
            if zig_file.stem not in ("types", "README"):
                # Convert file name to Python type name
                type_name = zig_file.stem.replace("py", "").replace("_", "")
                supported_types.add(type_name.lower())

        # Common Python types
        common_types = {
            "int", "float", "bool", "str", "bytes", "bytearray",
            "list", "tuple", "dict", "set", "frozenset",
            "complex", "range", "slice", "memoryview",
            "datetime", "date", "time", "timedelta",
            "decimal", "fraction", "uuid", "path",
            "deque", "defaultdict", "counter", "chainmap",
            "array"  # NumPy
        }

        missing_types = common_types - supported_types

        return {
            "supported": sorted(supported_types),
            "supported_count": len(supported_types),
            "missing": sorted(missing_types),
            "missing_count": len(missing_types),
            "coverage_percent": round(len(supported_types) / len(common_types) * 100, 1)
        }

    def generate_report(self) -> str:
        """Generate comprehensive diagnostic report."""
        lines = []
        lines.append("=" * 70)
        lines.append(" PYZ3 DIAGNOSTIC REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Environment
        lines.append("📋 ENVIRONMENT")
        lines.append("-" * 70)
        env = self.check_environment()
        for component, info in env.items():
            lines.append(f"  {component.replace('_', ' ').title()}: {info['status']}")
            if 'version' in info:
                lines.append(f"    Version: {info['version']}")
        lines.append("")

        # Configuration
        lines.append("⚙️  CONFIGURATION")
        lines.append("-" * 70)
        validation = self.validate_module_config()
        if validation['errors']:
            lines.append("  ❌ Errors:")
            for error in validation['errors']:
                lines.append(f"    - {error}")
        else:
            lines.append("  ✅ No configuration errors")

        if validation['warnings']:
            lines.append("  ⚠️  Warnings:")
            for warning in validation['warnings']:
                lines.append(f"    - {warning}")
        lines.append("")

        # Build artifacts
        lines.append("🔨 BUILD ARTIFACTS")
        lines.append("-" * 70)
        artifacts = self.analyze_build_artifacts()
        lines.append(f"  Compiled modules: {len(artifacts['modules'])}")
        lines.append(f"  Total size: {artifacts['total_size_mb']} MB")
        if artifacts['modules']:
            lines.append("  Modules:")
            for module in sorted(artifacts['modules'], key=lambda x: x['size'], reverse=True):
                lines.append(f"    - {module['name']}: {module['size_mb']} MB")
        if 'cache_size_mb' in artifacts:
            lines.append(f"  Cache size: {artifacts['cache_size_mb']} MB")
        lines.append("")

        # Type coverage
        lines.append("📊 TYPE COVERAGE")
        lines.append("-" * 70)
        coverage = self.check_type_coverage()
        if 'error' not in coverage:
            lines.append(f"  Coverage: {coverage['coverage_percent']}% "
                        f"({coverage['supported_count']}/{coverage['supported_count'] + coverage['missing_count']})")
            lines.append(f"  Supported types: {', '.join(coverage['supported'][:10])}")
            if len(coverage['supported']) > 10:
                lines.append(f"    ... and {len(coverage['supported']) - 10} more")
            if coverage['missing']:
                lines.append(f"  Missing types: {', '.join(coverage['missing'])}")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


def main():
    """CLI entry point for diagnostics."""
    import argparse

    parser = argparse.ArgumentParser(description="pyz3 diagnostic tools")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--environment", action="store_true", help="Check environment only")
    parser.add_argument("--config", action="store_true", help="Validate configuration only")
    parser.add_argument("--artifacts", action="store_true", help="Analyze build artifacts only")
    parser.add_argument("--coverage", action="store_true", help="Check type coverage only")

    args = parser.parse_args()
    diag = Diagnostics()

    if args.environment:
        result = diag.check_environment()
    elif args.config:
        result = diag.validate_module_config()
    elif args.artifacts:
        result = diag.analyze_build_artifacts()
    elif args.coverage:
        result = diag.check_type_coverage()
    else:
        # Full report
        print(diag.generate_report())
        return

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
