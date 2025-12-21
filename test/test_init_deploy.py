import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestInitCommand:

    def test_init_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyz3", "init", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "init" in result.stdout

    def test_init_in_temp_dir(self):
        """Test init command creates project files in current directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pyz3",
                    "init",
                    "-n",
                    "test_package",
                    "--no-interactive",
                ],
                cwd=tmppath,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Init failed. Output: {result.stdout}\nError: {result.stderr}"

            # Check that files were created in tmppath directly
            assert (tmppath / "pyproject.toml").exists(), "pyproject.toml not created"
            assert (tmppath / "test_package.zig").exists(), "test_package.zig not created"
            assert (tmppath / "README.md").exists(), "README.md not created"
            assert (tmppath / ".gitignore").exists(), ".gitignore not created"
            assert (tmppath / "test").exists(), "test directory not created"
            assert (tmppath / "test" / "test_test_package.py").exists(), "test file not created"

    def test_new_command(self):
        """Test new command creates project in a new subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pyz3",
                    "new",
                    "my_test_project",
                    "-p",
                    str(tmppath),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"New command failed. Output: {result.stdout}\nError: {result.stderr}"

            # Check that project directory was created
            project_path = tmppath / "my_test_project"
            assert project_path.exists(), "Project directory not created"
            assert (project_path / "pyproject.toml").exists(), "pyproject.toml not created"
            assert (project_path / "my_test_project.zig").exists(), "my_test_project.zig not created"
            assert (project_path / "README.md").exists(), "README.md not created"
            assert (project_path / ".gitignore").exists(), ".gitignore not created"
            assert (project_path / "test").exists(), "test directory not created"


class TestDeployCommand:

    def test_deploy_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyz3", "deploy", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "deploy" in result.stdout

    def test_deploy_without_dist_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pyz3",
                    "deploy",
                    "--dist-dir",
                    f"{tmpdir}/nonexistent",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode != 0
            combined = result.stdout + result.stderr
            assert "does not exist" in combined or "twine" in combined

    def test_deploy_empty_dist_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pyz3",
                    "deploy",
                    "--dist-dir",
                    tmpdir,
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode != 0
            combined = result.stdout + result.stderr
            assert "No wheel" in combined or "No distribution" in combined or "twine" in combined


class TestCheckCommand:

    def test_check_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyz3", "check", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "check" in result.stdout

    def test_check_without_dist_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pyz3",
                    "check",
                    "--dist-dir",
                    f"{tmpdir}/nonexistent",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode != 0 or "twine" in (result.stdout + result.stderr)


class TestProjectStructure:
    """Test the structure of generated projects."""

    def test_init_creates_valid_pyproject_toml(self):
        """Test that init creates a valid pyproject.toml with required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pyz3",
                    "init",
                    "-n",
                    "test_pkg",
                ],
                cwd=tmppath,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

            pyproject = tmppath / "pyproject.toml"
            assert pyproject.exists()

            content = pyproject.read_text()
            # Check for required sections
            assert "[build-system]" in content
            assert "[project]" in content
            assert "[tool.pyz3]" in content
            assert "pyz3" in content  # build backend
            assert "test_pkg" in content  # package name
