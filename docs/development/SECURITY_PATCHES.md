# Security Patches - Implementation Guide

This document provides specific code patches to fix critical security vulnerabilities in pyZ3.

## Critical Patch 1: Fix Command Injection in deps.py

### Location: `pyz3/deps.py` lines 165-195

**BEFORE (Vulnerable):**
```python
def _add_remote_dependency(
    self, url: str, name: Optional[str], verbose: bool
) -> Dependency:
    """Add a dependency from a remote URL."""
    print(f"  [1/4] Cloning from {url}...")

    if name is None:
        name = Path(urlparse(url).path).stem
        name = name.replace("-", "_").replace(".", "_")

    self.deps_dir.mkdir(exist_ok=True)
    dep_path = self.deps_dir / name

    if dep_path.exists():
        print(f"  ⚠️  Directory {dep_path} already exists, using existing...")
    else:
        try:
            subprocess.run(
                ["git", "clone", url, str(dep_path)],  # VULNERABLE!
                check=True,
                capture_output=not verbose,
            )
            print(f"  ✓ Cloned to deps/{name}")
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed to clone repository")
            sys.exit(1)
```

**AFTER (Secure):**
```python
def _add_remote_dependency(
    self, url: str, name: Optional[str], verbose: bool
) -> Dependency:
    """Add a dependency from a remote URL."""
    from pyz3.security import SecurityValidator, SecurityError
    import logging

    logger = logging.getLogger(__name__)

    # SECURITY: Validate URL before using
    is_valid, error = SecurityValidator.validate_git_url(url)
    if not is_valid:
        logger.error(f"Invalid Git URL: {error}")
        raise SecurityError(f"Security validation failed: {error}")

    print(f"  [1/4] Cloning from {url}...")

    if name is None:
        name = Path(urlparse(url).path).stem
        name = name.replace("-", "_").replace(".", "_")

    # SECURITY: Validate package name
    is_valid, error, name = SecurityValidator.sanitize_package_name(name)
    if not is_valid:
        logger.error(f"Invalid package name: {error}")
        raise SecurityError(f"Invalid package name: {error}")

    self.deps_dir.mkdir(exist_ok=True)
    dep_path = self.deps_dir / name

    if dep_path.exists():
        print(f"  ⚠️  Directory {dep_path} already exists, using existing...")
    else:
        try:
            # SECURITY: Safe git clone with timeouts and disabled hooks
            result = subprocess.run(
                [
                    "git", "clone",
                    "--depth=1",  # Shallow clone
                    "--config", "core.hooksPath=/dev/null",  # Disable hooks
                    "--config", "core.fsmonitor=false",  # Disable fsmonitor
                    url,
                    str(dep_path)
                ],
                check=True,
                capture_output=not verbose,
                timeout=300,  # 5 minute timeout
                text=True,
            )
            print(f"  ✓ Cloned to deps/{name}")

            # SECURITY: Verify cloned repository
            hooks = SecurityValidator.scan_for_git_hooks(dep_path)
            if hooks:
                logger.warning(f"Git hooks found: {', '.join(hooks)}")

            is_size_ok, size = SecurityValidator.check_directory_size(dep_path)
            if not is_size_ok:
                logger.warning(f"Repository is large: {size // 1_000_000}MB")

        except subprocess.TimeoutExpired:
            logger.error("Git clone timeout after 5 minutes")
            # Clean up partial clone
            if dep_path.exists():
                import shutil
                shutil.rmtree(dep_path)
            raise SecurityError("Git clone timed out")

        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed: {e}")
            print(f"  ❌ Failed to clone repository")
            if e.stderr:
                logger.debug(f"Git error: {e.stderr}")
            sys.exit(1)
```

---

## Critical Patch 2: Fix Path Traversal in deps.py

### Location: `pyz3/deps.py` lines 220-240

**BEFORE (Vulnerable):**
```python
def _add_local_dependency(
    self, path: str, name: Optional[str], verbose: bool
) -> Dependency:
    """Add a dependency from a local path."""
    local_path = Path(path).resolve()  # VULNERABLE!

    if not local_path.exists():
        print(f"  ❌ Path does not exist: {local_path}")
        sys.exit(1)

    print(f"  [1/4] Using local library at {local_path}...")

    if name is None:
        name = local_path.stem.replace("-", "_").replace(".", "_")
```

**AFTER (Secure):**
```python
def _add_local_dependency(
    self, path: str, name: Optional[str], verbose: bool
) -> Dependency:
    """Add a dependency from a local path."""
    from pyz3.security import SecurityValidator, SecurityError
    import logging

    logger = logging.getLogger(__name__)

    # SECURITY: Validate local path
    is_valid, error, local_path = SecurityValidator.validate_local_path(
        path, self.project_root
    )
    if not is_valid:
        logger.error(f"Invalid path: {error}")
        raise SecurityError(f"Path validation failed: {error}")

    print(f"  [1/4] Using local library at {local_path.name}...")  # Only show basename

    if name is None:
        name = local_path.stem.replace("-", "_").replace(".", "_")

    # SECURITY: Validate package name
    is_valid, error, name = SecurityValidator.sanitize_package_name(name)
    if not is_valid:
        logger.error(f"Invalid package name: {error}")
        raise SecurityError(f"Invalid package name: {error}")
```

---

## Critical Patch 3: Fix Unsafe File Writes in init.py

### Location: `pyz3/init.py` lines 336-370

**BEFORE (Vulnerable):**
```python
# Create __init__.py for the package
init_py = path / package_name / "__init__.py"
if not init_py.exists() or force:
    init_py.write_text(f'"""The {package_name} package."""\n\n__version__ = "0.1.0"\n')
    print(f"  ✓ Created {init_py.relative_to(path)}")

# Create pyproject.toml
pyproject_path = path / "pyproject.toml"
if not pyproject_path.exists() or force:
    pyproject_path.write_text(
        TEMPLATE_PYPROJECT.format(
            package_name=package_name,
            module_name=module_name,
            author=author,
        )
    )
    print(f"  ✓ Created {pyproject_path.relative_to(path)}")
```

**AFTER (Secure):**
```python
from pyz3.security import SecurityValidator, SecurityError

# Create __init__.py for the package
init_py = path / package_name / "__init__.py"
if not init_py.exists() or force:
    content = f'"""The {package_name} package."""\n\n__version__ = "0.1.0"\n'
    try:
        SecurityValidator.safe_write_text(init_py, content, force=force)
        print(f"  ✓ Created {init_py.relative_to(path)}")
    except (SecurityError, IOError) as e:
        print(f"  ❌ Failed to create {init_py.name}: {e}")
        return

# Create pyproject.toml
pyproject_path = path / "pyproject.toml"
if not pyproject_path.exists() or force:
    # SECURITY: Escape author string for TOML
    author_safe = SecurityValidator.escape_toml_string(author)

    content = TEMPLATE_PYPROJECT.format(
        package_name=package_name,
        module_name=module_name,
        author=author_safe,
    )

    try:
        SecurityValidator.safe_write_text(pyproject_path, content, force=force)
        print(f"  ✓ Created {pyproject_path.relative_to(path)}")
    except (SecurityError, IOError) as e:
        print(f"  ❌ Failed to create {pyproject_path.name}: {e}")
        return
```

---

## High Priority Patch 4: Fix Broad Exception Handling in deps.py

### Location: `pyz3/deps.py` lines 195-207

**BEFORE (Problematic):**
```python
# Try to detect version from git
version = None
try:
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        cwd=dep_path,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        version = result.stdout.strip()
except:  # DANGEROUS! Catches EVERYTHING
    pass
```

**AFTER (Proper):**
```python
# Try to detect version from git
version = None
try:
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        cwd=dep_path,
        capture_output=True,
        text=True,
        timeout=10,  # Add timeout
        check=False,  # Don't raise on non-zero
    )
    if result.returncode == 0:
        version = result.stdout.strip()
        # Sanitize version string
        if len(version) > 50:
            version = version[:50]
except subprocess.TimeoutExpired:
    logger.debug("Git version detection timed out")
except subprocess.SubprocessError as e:
    logger.debug(f"Could not detect git version: {e}")
except Exception as e:
    # Unexpected errors - log but don't fail
    logger.warning(f"Unexpected error detecting version: {e}")
```

---

## High Priority Patch 5: Add Subprocess Timeouts in develop.py

### Location: `pyz3/develop.py` lines 32-48

**BEFORE (No Timeout):**
```python
buildzig.zig_build(
    argv=["install", f"-Dpython-exe={sys.executable}", f"-Doptimize={optimize}"],
    conf=conf,
    env=env,
)
```

**AFTER (With Timeout):**
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Build timeout")

# Set timeout for build (e.g., 10 minutes)
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(600)  # 10 minutes

try:
    buildzig.zig_build(
        argv=["install", f"-Dpython-exe={sys.executable}", f"-Doptimize={optimize}"],
        conf=conf,
        env=env,
    )
finally:
    signal.alarm(0)  # Cancel timeout
```

---

## Medium Priority Patch 6: Fix Information Disclosure

### Location: Multiple files

**Pattern to Replace:**
```python
print(f"  ❌ Error: {absolute_path}")  # Leaks full path
```

**Replace With:**
```python
# Only show relative or sanitized paths
safe_path = absolute_path.relative_to(project_root) if in_project else absolute_path.name
print(f"  ❌ Error: {safe_path}")
```

---

## Implementation Checklist

### Step 1: Add Security Module
- [x] Create `pyz3/security.py`
- [ ] Add unit tests for `SecurityValidator`
- [ ] Test on Windows, Linux, macOS

### Step 2: Update deps.py
- [ ] Apply Patch 1 (Command injection)
- [ ] Apply Patch 2 (Path traversal)
- [ ] Apply Patch 4 (Exception handling)
- [ ] Add logging throughout
- [ ] Test with various malicious inputs

### Step 3: Update init.py
- [ ] Apply Patch 3 (File writes)
- [ ] Fix TOCTOU race conditions
- [ ] Add logging
- [ ] Test edge cases

### Step 4: Update develop.py
- [ ] Apply Patch 5 (Timeouts)
- [ ] Improve error handling
- [ ] Add validation

### Step 5: Testing
- [ ] Write security-focused tests
- [ ] Test command injection attempts
- [ ] Test path traversal attempts
- [ ] Test on all platforms
- [ ] Fuzzing tests

### Step 6: Documentation
- [ ] Update README with security notes
- [ ] Add security best practices guide
- [ ] Document trusted git hosts
- [ ] Explain validation rules

---

## Testing the Patches

### Security Test Suite

```python
# test_security_patches.py
import pytest
from pyz3.security import SecurityValidator, SecurityError

def test_command_injection_prevention():
    """Test that command injection is prevented."""
    malicious_urls = [
        "https://github.com/user/repo; rm -rf /",
        "https://github.com/user/repo && curl evil.com",
        "https://github.com/user/repo | bash",
    ]

    for url in malicious_urls:
        is_valid, error = SecurityValidator.validate_git_url(url)
        assert not is_valid, f"Should reject: {url}"

def test_path_traversal_prevention():
    """Test that path traversal is prevented."""
    malicious_paths = [
        "../../../etc/passwd",
        "../../.ssh/id_rsa",
        "/etc/shadow",
    ]

    project_root = Path("/home/user/project")
    for path in malicious_paths:
        is_valid, error, _ = SecurityValidator.validate_local_path(path, project_root)
        assert not is_valid, f"Should reject: {path}"

def test_symlink_attack_prevention(tmp_path):
    """Test that symlink attacks are prevented."""
    # Create a symlink to /etc/passwd
    symlink = tmp_path / "malicious_link"
    symlink.symlink_to("/etc/passwd")

    with pytest.raises(SecurityError):
        SecurityValidator.safe_write_text(symlink, "malicious content")

def test_package_name_validation():
    """Test package name sanitization."""
    test_cases = [
        ("valid_name", True),
        ("", False),  # Empty
        ("class", True),  # Keyword - should add suffix
        ("os", False),  # System module
        ("a" * 101, False),  # Too long
    ]

    for name, should_pass in test_cases:
        is_valid, error, sanitized = SecurityValidator.sanitize_package_name(name)
        assert is_valid == should_pass, f"Failed for: {name}"
```

---

## Rollout Plan

### Phase 1: Critical Fixes (Week 1)
1. Merge security.py module
2. Apply command injection fixes
3. Apply path traversal fixes
4. Apply file write fixes
5. Basic security testing

### Phase 2: High Priority (Week 2)
6. Fix exception handling
7. Add timeouts
8. Add logging infrastructure
9. Extended security testing

### Phase 3: Medium Priority (Week 3)
10. Fix information disclosure
11. Add input validation everywhere
12. Platform compatibility testing
13. Documentation updates

### Phase 4: Code Quality (Week 4)
14. Performance optimizations
15. Add type hints
16. Comprehensive test suite
17. Security audit

---

## Verification

After applying patches, verify with:

```bash
# 1. Run security test suite
pytest test_security_patches.py -v

# 2. Try malicious inputs
pyz3 add "https://evil.com/repo"  # Should fail
pyz3 add "../../../etc/passwd"  # Should fail

# 3. Check generated code
pyz3 new test_project
# Verify no symlinks are followed
# Verify files are written safely

# 4. Security scan
bandit -r pyz3/
safety check

# 5. Code quality
mypy pyz3/
ruff check pyz3/
```

---

## Conclusion

These patches address the most critical security vulnerabilities found in the analysis. Implementing them will significantly improve the security posture of pyZ3.

**Priority Order:**
1. **Critical** - Apply immediately before any production use
2. **High** - Apply within 1 week
3. **Medium** - Apply within 2-3 weeks
4. **Code Quality** - Ongoing improvements

Each patch is focused, testable, and doesn't break existing functionality when applied correctly.
