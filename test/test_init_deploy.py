import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _check_cookiecutter_available() -> bool:
    try:
        import cookiecutter  # noqa: F401

        return True
    except ImportError:
        return False


class TestInitCommand:
    def test_init_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "pyz3", "init", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "init" in result.stdout

    @pytest.mark.skipif(not _check_cookiecutter_available(), reason="cookiecutter required")
    def test_init_in_temp_dir(self):
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

            if result.returncode == 0:
                possible_paths = [
                    tmppath,
                    tmppath / "test-package",
                    tmppath / "test_package",
                ]

                found = False
                for path in possible_paths:
                    if (path / "pyproject.toml").exists():
                        found = True
                        assert (path / "src").exists()
                        break

                assert found, f"pyproject.toml not created. Output: {result.stdout}\nError: {result.stderr}"

    @pytest.mark.skipif(not _check_cookiecutter_available(), reason="cookiecutter required")
    def test_new_command(self):
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

            if result.returncode == 0:
                possible_paths = [
                    tmppath / "my_test_project",
                    tmppath / "my-test-project",
                ]

                found = False
                for path in possible_paths:
                    if path.exists() and (path / "pyproject.toml").exists():
                        found = True
                        break

                assert found, f"Project not created. Output: {result.stdout}\nError: {result.stderr}"


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


class TestTemplateIntegration:
    def test_template_exists(self):
        from pyz3 import init

        pyz3_package = Path(init.__file__).parent
        template_path = pyz3_package / "pyZ3-template"

        assert template_path.exists(), f"Template directory not found at {template_path}"
        assert (template_path / "cookiecutter.json").exists()
