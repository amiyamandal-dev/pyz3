"""
Development installation utilities for pyz3.

Similar to Maturin's develop functionality, this module provides
utilities for building and installing pyz3 projects in development mode.

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

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from pyz3.logging_config import get_logger

logger = get_logger(__name__)


def _find_installer() -> tuple[list[str], str]:
    """
    Find the best available package installer (uv or pip).

    Returns:
        Tuple of (command list, installer name)
    """
    # Prefer uv if available
    if shutil.which("uv"):
        return ["uv", "pip"], "uv"
    return [sys.executable, "-m", "pip"], "pip"


def develop_install(
    optimize: str = "Debug",
    verbose: bool = False,
    extras: Optional[list[str]] = None,
) -> None:
    """
    Build the extension and install the package in development mode.

    This is similar to `pip install -e .` or `uv pip install -e .` but also
    builds the Zig extension modules first.

    Args:
        optimize: Optimization level (Debug, ReleaseSafe, ReleaseFast, ReleaseSmall)
        verbose: Enable verbose output
        extras: Optional list of extras to install
    """
    logger.info(f"Starting development installation (optimize={optimize})")

    project_root = Path.cwd()
    pyproject_path = project_root / "pyproject.toml"

    # Validate project structure
    if not pyproject_path.exists():
        logger.error("pyproject.toml not found in current directory")
        print("Error: pyproject.toml not found in current directory!")
        print("   Make sure you're in a pyz3 project directory.")
        sys.exit(1)

    # Validate pyproject.toml is readable
    try:
        pyproject_path.stat()
        logger.debug(f"Found pyproject.toml at {pyproject_path}")
    except (OSError, PermissionError) as e:
        logger.error(f"Cannot access pyproject.toml: {e}")
        print(f"Error: Cannot access pyproject.toml: {e}")
        sys.exit(1)

    print(f"Building and installing in development mode (optimize={optimize})...")

    # Step 1: Build the extension modules using pyz3 build command
    print("\n[1/3] Building Zig extension modules...")
    try:
        from pyz3 import buildzig, config

        logger.debug("Loading pyz3 configuration")
        conf = config.load()

        env = os.environ.copy()
        env["PYZ3_OPTIMIZE"] = optimize

        logger.debug(f"Running zig build with optimize={optimize}")
        buildzig.zig_build(
            argv=["install", f"-Dpython-exe={sys.executable}", f"-Doptimize={optimize}"],
            conf=conf,
            env=env,
        )
        print("  [OK] Extension modules built successfully")
        logger.info("Zig extension modules built successfully")
    except ImportError as e:
        logger.error(f"Failed to import pyz3 modules: {e}")
        print(f"  [FAIL] Failed to import pyz3 modules: {e}")
        print("     Make sure pyz3 is installed correctly.")
        sys.exit(1)
    except subprocess.TimeoutExpired as e:
        logger.error("Build timeout expired")
        print(f"  [FAIL] Build timed out after {e.timeout} seconds")
        print("     The build process took too long. Check for infinite loops or hanging processes.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed with exit code {e.returncode}")
        print(f"  [FAIL] Build failed with exit code {e.returncode}")
        if verbose and e.stderr:
            print(f"     {e.stderr}")
        sys.exit(1)
    except (OSError, PermissionError) as e:
        logger.error(f"Build failed due to OS error: {e}")
        print(f"  [FAIL] Build failed: {e}")
        print("     Check file permissions and disk space.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected build error: {e}")
        print(f"  [FAIL] Failed to build extension modules: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Step 2: Install the package in editable mode
    print("\n[2/3] Installing package in editable mode...")

    installer_cmd, installer_name = _find_installer()
    install_target = "."

    if extras:
        extras_str = ",".join(extras)
        install_target = f".[{extras_str}]"
        logger.debug(f"Installing with extras: {extras_str}")

    pip_cmd = installer_cmd + ["install", "-e", install_target]

    if verbose:
        pip_cmd.append("-v")

    try:
        logger.debug(f"Running: {' '.join(pip_cmd)}")
        result = subprocess.run(
            pip_cmd,
            cwd=project_root,
            check=True,
            capture_output=not verbose,
            text=True,
            timeout=600,  # 10 minute timeout for install
        )
        print(f"  [OK] Package installed in editable mode (using {installer_name})")
        logger.info(f"Package installed in editable mode using {installer_name}")
    except subprocess.TimeoutExpired:
        logger.error("Install timed out after 10 minutes")
        print("  [FAIL] Install timed out after 10 minutes")
        print("     The installation took too long. Check network connectivity or dependencies.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Install failed with exit code {e.returncode}")
        print(f"  [FAIL] Failed to install package (exit code {e.returncode})")
        if not verbose and e.stderr:
            print(f"     {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("Installer not found")
        print("  [FAIL] Error: Package installer not found")
        print(f"     Could not run: {' '.join(installer_cmd)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during install: {e}")
        print(f"  [FAIL] Unexpected error during install: {e}")
        sys.exit(1)

    # Step 3: Verify installation
    print("\n[3/3] Verifying installation...")
    try:
        import tomllib

        logger.debug("Reading pyproject.toml for verification")
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)

        # Try different locations for package name
        package_name = None
        if "project" in pyproject:
            package_name = pyproject["project"].get("name")
        if not package_name and "tool" in pyproject and "poetry" in pyproject["tool"]:
            package_name = pyproject["tool"]["poetry"].get("name")

        if package_name:
            logger.debug(f"Package name from pyproject.toml: {package_name}")
            # Try to import the package
            try:
                __import__(package_name.replace("-", "_"))
                print(f"  [OK] Package '{package_name}' is importable")
                logger.info(f"Verified package {package_name} is importable")
            except ImportError as e:
                logger.warning(f"Package not importable: {e}")
                print(f"  [WARN] Warning: Could not import '{package_name}': {e}")
                print("     The package is installed but may have import issues.")
        else:
            logger.warning("Could not determine package name")
            print("  [WARN] Could not determine package name from pyproject.toml")

    except tomllib.TOMLDecodeError as e:
        logger.error(f"Invalid TOML in pyproject.toml: {e}")
        print(f"  [WARN] Warning: Invalid TOML in pyproject.toml: {e}")
    except KeyError as e:
        logger.error(f"Missing key in pyproject.toml: {e}")
        print(f"  [WARN] Warning: Could not find package name in pyproject.toml: {e}")
    except (OSError, PermissionError) as e:
        logger.error(f"Cannot read pyproject.toml: {e}")
        print(f"  [WARN] Warning: Could not read pyproject.toml: {e}")
    except Exception as e:
        logger.warning(f"Verification failed: {e}")
        print(f"  [WARN] Warning: Could not verify installation: {e}")

    logger.info("Development installation complete")
    print("\n[DONE] Development installation complete!\n")
    print("You can now:")
    print("  - Import your package in Python")
    print("  - Run tests with: pytest")
    print("  - Make changes to Zig code and rebuild with: pyz3 develop")


def develop_build_only(optimize: str = "Debug", verbose: bool = False) -> None:
    """
    Build the extension modules without installing.

    Args:
        optimize: Optimization level
        verbose: Enable verbose output
    """
    logger.info(f"Starting build-only mode (optimize={optimize})")

    project_root = Path.cwd()
    pyproject_path = project_root / "pyproject.toml"

    # Validate project structure
    if not pyproject_path.exists():
        logger.error("pyproject.toml not found in current directory")
        print("Error: pyproject.toml not found in current directory!")
        print("   Make sure you're in a pyz3 project directory.")
        sys.exit(1)

    # Validate pyproject.toml is readable
    try:
        pyproject_path.stat()
        logger.debug(f"Found pyproject.toml at {pyproject_path}")
    except (OSError, PermissionError) as e:
        logger.error(f"Cannot access pyproject.toml: {e}")
        print(f"Error: Cannot access pyproject.toml: {e}")
        sys.exit(1)

    print(f"Building Zig extension modules (optimize={optimize})...")

    try:
        from pyz3 import buildzig, config

        logger.debug("Loading pyz3 configuration")
        conf = config.load()

        env = os.environ.copy()
        env["PYZ3_OPTIMIZE"] = optimize

        logger.debug(f"Running zig build with optimize={optimize}")
        buildzig.zig_build(
            argv=["install", f"-Dpython-exe={sys.executable}", f"-Doptimize={optimize}"],
            conf=conf,
            env=env,
        )
        print("[DONE] Extension modules built successfully!")
        logger.info("Build-only mode completed successfully")
    except ImportError as e:
        logger.error(f"Failed to import pyz3 modules: {e}")
        print(f"[FAIL] Failed to import pyz3 modules: {e}")
        print("   Make sure pyz3 is installed correctly.")
        sys.exit(1)
    except subprocess.TimeoutExpired as e:
        logger.error("Build timeout expired")
        print(f"[FAIL] Build timed out after {e.timeout} seconds")
        print("   The build process took too long. Check for infinite loops or hanging processes.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Build failed with exit code {e.returncode}")
        print(f"[FAIL] Build failed with exit code {e.returncode}")
        if verbose and e.stderr:
            print(f"   {e.stderr}")
        sys.exit(1)
    except (OSError, PermissionError) as e:
        logger.error(f"Build failed due to OS error: {e}")
        print(f"[FAIL] Build failed: {e}")
        print("   Check file permissions and disk space.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected build error: {e}")
        print(f"[FAIL] Failed to build extension modules: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
