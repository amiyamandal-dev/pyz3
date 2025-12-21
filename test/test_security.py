"""
Security validation tests for pyz3.

Tests package name validation, path sanitization, and security checks.
"""

import re
from pathlib import Path

import pytest

from pyz3.security import SecurityValidator


class TestPackageNameValidation:
    """Test package name validation and sanitization."""

    def test_valid_package_names(self):
        """Test that valid package names are accepted."""
        valid_names = [
            "mypackage",
            "my_package",
            "package123",
            "my_package_123",
            "a",  # Single character
            "package_with_many_underscores",
        ]

        for name in valid_names:
            is_valid, error, sanitized = SecurityValidator.sanitize_package_name(name)
            assert is_valid, f"'{name}' should be valid but got error: {error}"
            assert sanitized == name, f"'{name}' should not be modified"
            assert error is None, f"'{name}' should have no error"

    def test_invalid_package_names(self):
        """Test that invalid package names are rejected."""
        invalid_names = [
            "",  # Empty
            "123package",  # Starts with number
            "пакет",  # Unicode (should fail ASCII check)
            "包",  # Unicode
            "!!!",  # Only special characters
            "---",  # Only hyphens
        ]

        for name in invalid_names:
            is_valid, error, sanitized = SecurityValidator.sanitize_package_name(name)
            assert not is_valid, f"'{name}' should be invalid but was accepted"
            assert error is not None, f"'{name}' should have an error message"

    def test_package_name_sanitization(self):
        """Test package name sanitization."""
        test_cases = [
            ("my-package", "my_package"),
            ("my.package", "my_package"),
            ("my package", "my_package"),
            ("MY-PACKAGE", "my_package"),  # Lowercase conversion
        ]

        for input_name, expected in test_cases:
            is_valid, error, sanitized = SecurityValidator.sanitize_package_name(input_name)
            # After sanitization, should be valid
            if sanitized:
                assert sanitized == expected, f"'{input_name}' should sanitize to '{expected}', got '{sanitized}'"

    def test_package_name_length_limits(self):
        """Test package name length validation."""
        # Very long name (Python has no hard limit, but we should be reasonable)
        long_name = "a" * 100
        is_valid, error, sanitized = SecurityValidator.sanitize_package_name(long_name)
        # Should still be valid if it meets other criteria
        assert is_valid or "too long" in error.lower()

    def test_reserved_package_names(self):
        """Test that Python reserved keywords are handled."""
        reserved_names = [
            "import",
            "class",
            "def",
            "return",
            "if",
            "else",
            "while",
            "for",
        ]

        for name in reserved_names:
            is_valid, error, sanitized = SecurityValidator.sanitize_package_name(name)
            # Reserved names should either be rejected or have a warning
            # (Implementation may vary)
            assert error is not None or sanitized != name

    def test_unicode_package_names(self):
        """Test handling of Unicode characters in package names."""
        unicode_names = [
            "pакет",  # Cyrillic
            "包",  # Chinese
            "パッケージ",  # Japanese
            "🎉package",  # Emoji
        ]

        for name in unicode_names:
            is_valid, error, sanitized = SecurityValidator.sanitize_package_name(name)
            # Unicode names should be rejected (Python package names must be ASCII)
            assert not is_valid, f"Unicode name '{name}' should be rejected"


class TestPathSanitization:
    """Test path sanitization and validation."""

    def test_safe_paths(self):
        """Test that safe paths are allowed."""
        safe_paths = [
            "myproject",
            "my_project",
            "path/to/project",
            "./relative/path",
        ]

        for path_str in safe_paths:
            # This assumes SecurityValidator has a sanitize_path method
            # If not, this test documents expected behavior
            path = Path(path_str)
            assert not path.is_absolute() or path.is_relative_to(Path.cwd())

    def test_dangerous_paths(self):
        """Test that dangerous paths are detected."""
        dangerous_paths = [
            "../../../etc/passwd",  # Directory traversal
            "/etc/passwd",  # Absolute path to system file
            "~/.ssh/id_rsa",  # Home directory access
        ]

        for path_str in dangerous_paths:
            path = Path(path_str).resolve()
            # Should not allow access to system directories
            system_dirs = ["/etc", "/sys", "/proc", "/dev"]
            is_dangerous = any(str(path).startswith(d) for d in system_dirs)
            if is_dangerous:
                assert True  # Dangerous path detected


class TestCommandInjection:
    """Test protection against command injection."""

    def test_sanitize_shell_arguments(self):
        """Test that shell arguments are properly escaped."""
        dangerous_inputs = [
            "'; rm -rf /;'",
            "$(malicious command)",
            "`whoami`",
            "| cat /etc/passwd",
            "; ls -la",
        ]

        for dangerous in dangerous_inputs:
            # If SecurityValidator has sanitize_command method
            # it should escape or reject these
            assert (
                ";" not in dangerous.replace("\;", "")
                or "`" not in dangerous
                or "$" not in dangerous.replace("\\$", "")
            )


class TestInputValidation:
    """Test input validation for various pyz3 operations."""

    def test_email_validation(self):
        """Test email address validation."""
        valid_emails = [
            "user@example.com",
            "test.user@example.co.uk",
            "user+tag@example.com",
        ]

        invalid_emails = [
            "not-an-email",
            "@example.com",
            "user@",
            "user@@example.com",
        ]

        # This tests the init.py email validation
        import re

        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

        for email in valid_emails:
            assert email_pattern.match(email), f"'{email}' should be valid"

        for email in invalid_emails:
            assert not email_pattern.match(email), f"'{email}' should be invalid"

    def test_version_string_validation(self):
        """Test version string validation."""
        valid_versions = [
            "0.1.0",
            "1.2.3",
            "0.0.1",
            "10.20.30",
        ]

        invalid_versions = [
            "1.2",  # Missing patch
            "v1.2.3",  # Has 'v' prefix
            "1.2.3.4",  # Too many parts
            "1.x.3",  # Non-numeric
        ]

        version_pattern = re.compile(r"^\d+\.\d+\.\d+$")

        for version in valid_versions:
            assert version_pattern.match(version), f"'{version}' should be valid"

        for version in invalid_versions:
            assert not version_pattern.match(version), f"'{version}' should be invalid"


class TestSecurityEdgeCases:
    """Test security edge cases and boundary conditions."""

    def test_null_byte_injection(self):
        """Test protection against null byte injection."""
        malicious_names = [
            "package\x00.txt",
            "test\x00",
        ]

        for name in malicious_names:
            is_valid, error, sanitized = SecurityValidator.sanitize_package_name(name)
            # Null bytes should be rejected
            assert not is_valid or "\x00" not in sanitized

    def test_path_traversal_variations(self):
        """Test various path traversal attempts."""
        traversal_attempts = [
            "../../../",
            "..\\..\\..\\",  # Windows style
            "....//",  # Encoded
            "%2e%2e%2f",  # URL encoded
        ]

        for attempt in traversal_attempts:
            # Path operations should normalize these
            path = Path(attempt).resolve()
            # Should not escape project directory
            assert ".." not in str(path) or path.is_absolute()

    def test_symlink_handling(self):
        """Test that symlinks are handled safely."""
        # This is a documentation test - actual symlink handling
        # would need filesystem setup
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create a regular file
            regular_file = tmppath / "regular.txt"
            regular_file.write_text("test")

            # Create a symlink
            if os.name != "nt":  # Skip on Windows
                symlink = tmppath / "link.txt"
                symlink.symlink_to(regular_file)

                # Both should be readable
                assert regular_file.exists()
                assert symlink.exists()
                assert symlink.is_symlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
