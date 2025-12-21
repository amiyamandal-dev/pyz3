# Version Management Guide

This guide explains the different ways to bump versions in the pyz3 project.

## Quick Reference

| Method | Command | Best For |
|--------|---------|----------|
| **Makefile** | `make bump-patch` | Quick, convenient version bumps |
| **Bash Script** | `./bump_version.sh patch` | Full control with detailed output |
| **Poetry Direct** | `poetry version patch` | Manual control (requires updating `__init__.py` manually) |

## Recommended Approach: Makefile

The **Makefile** is the easiest and most convenient method for common version bumps.

### Usage

```bash
# Show current version
make version

# Patch release (0.1.0 → 0.1.1) - for bug fixes
make bump-patch

# Minor release (0.1.0 → 0.2.0) - for new features
make bump-minor

# Major release (0.1.0 → 1.0.0) - for breaking changes
make bump-major

# Pre-releases
make bump-prepatch      # 0.1.0 → 0.1.1-alpha.0
make bump-preminor      # 0.1.0 → 0.2.0-alpha.0
make bump-premajor      # 0.1.0 → 1.0.0-alpha.0
make bump-prerelease    # 0.1.1-alpha.0 → 0.1.1-alpha.1
```

### Advantages
- ✅ Shortest commands
- ✅ Easy to remember
- ✅ Automatically updates both `pyproject.toml` and `pyz3/__init__.py`
- ✅ Provides clear next steps

---

## Advanced Approach: Bash Script

The **bash script** (`bump_version.sh`) provides the most flexibility and detailed output.

### Usage

```bash
# Using bump rules (recommended)
./bump_version.sh patch        # 0.1.0 → 0.1.1
./bump_version.sh minor        # 0.1.0 → 0.2.0
./bump_version.sh major        # 0.1.0 → 1.0.0

# Using explicit versions
./bump_version.sh 0.2.0        # Set to specific version
./bump_version.sh 0.2.0-beta.1 # Set to specific pre-release

# Dry run (preview changes without making them)
./bump_version.sh patch --dry-run
./bump_version.sh 0.2.0 --dry-run

# Show help
./bump_version.sh --help
```

### Features
- ✅ Dry-run mode to preview changes
- ✅ Detailed output with color coding
- ✅ Automatic validation
- ✅ Updates `pyproject.toml`, `pyz3/__init__.py`, and `poetry.lock`
- ✅ Provides git commands for next steps

### Example Output

```
Current version: 0.1.0
Bumping version with poetry... ✓
New version:     0.1.1

Updating pyz3/__init__.py... ✓
Updating poetry.lock... ✓

✅ Version successfully updated to 0.1.1

Next steps:
  1. Review changes: git diff
  2. Commit changes:
     git add pyproject.toml pyz3/__init__.py poetry.lock
     git commit -m "Bump version to 0.1.1"
  3. Create git tag:
     git tag v0.1.1
  4. Push changes:
     git push origin main --tags

After pushing:
  ✓ GitHub Actions will build wheels for all platforms
  ✓ Create a GitHub release from tag v0.1.1
  ✓ Wheels will be uploaded to release (if configured)
  ✓ Publish to PyPI (if configured)

Estimated time: ~10 minutes
```

---

## Manual Approach: Poetry Direct

You can use Poetry directly, but you'll need to manually update `pyz3/__init__.py`.

### Usage

```bash
# Bump version in pyproject.toml
poetry version patch  # or minor, major, etc.

# Get the new version
NEW_VERSION=$(poetry version -s)

# Manually update __init__.py
sed -i '' "s/^__version__ = .*/__version__ = \"$NEW_VERSION\"/" pyz3/__init__.py

# Update lock file
poetry lock --no-update
```

### When to Use
- When you want direct control
- When integrating into custom automation
- When the scripts are not available

---

## Semantic Versioning Rules

pyz3 follows [Semantic Versioning](https://semver.org/):

### Version Format: `MAJOR.MINOR.PATCH`

- **MAJOR** (1.0.0): Breaking changes - incompatible API changes
- **MINOR** (0.2.0): New features - backward-compatible functionality
- **PATCH** (0.1.1): Bug fixes - backward-compatible fixes

### Pre-release Versions

Format: `MAJOR.MINOR.PATCH-PRERELEASE.NUMBER`

Examples:
- `0.2.0-alpha.0` - First alpha release of 0.2.0
- `0.2.0-beta.1` - Second beta release of 0.2.0
- `1.0.0-rc.2` - Third release candidate for 1.0.0

### Bump Rules

| Rule | From | To | Use Case |
|------|------|-----|----------|
| `patch` | 0.1.0 | 0.1.1 | Bug fix release |
| `minor` | 0.1.0 | 0.2.0 | New feature release |
| `major` | 0.1.0 | 1.0.0 | Breaking change release |
| `prepatch` | 0.1.0 | 0.1.1-alpha.0 | Start patch pre-release |
| `preminor` | 0.1.0 | 0.2.0-alpha.0 | Start minor pre-release |
| `premajor` | 0.1.0 | 1.0.0-alpha.0 | Start major pre-release |
| `prerelease` | 0.2.0-alpha.0 | 0.2.0-alpha.1 | Increment pre-release |

---

## Complete Release Workflow

Here's the recommended workflow for creating a new release:

### 1. Bump the Version

```bash
# For a patch release (bug fixes)
make bump-patch

# Or use the script for more control
./bump_version.sh patch
```

### 2. Review Changes

```bash
git diff
```

Verify that both files are updated:
- `pyproject.toml`
- `pyz3/__init__.py`

### 3. Commit Changes

```bash
git add pyproject.toml pyz3/__init__.py pyz3/pyZ3-template/cookiecutter.json poetry.lock
git commit -m "Bump version to 0.1.1"
```

### 4. Create Git Tag

```bash
git tag v0.1.1
```

### 5. Push to GitHub

```bash
git push origin main --tags
```

### 6. GitHub Actions Takes Over

After pushing:
1. **CI workflow** runs tests on all platforms
2. **Build workflow** creates wheels for 12 platforms
3. Wheels are attached to the GitHub release
4. (Optional) Package is published to PyPI

### 7. Create GitHub Release

1. Go to https://github.com/yourusername/pyz3/releases
2. Click "Draft a new release"
3. Select tag `v0.1.1`
4. Add release notes (changelog, features, fixes)
5. Publish release

---

## Files Updated by Version Bump

When you bump the version, these files are automatically updated:

| File | Updated By | Example |
|------|------------|---------|
| `pyproject.toml` | Poetry | `version = "0.1.1"` |
| `pyz3/__init__.py` | Script | `__version__ = "0.1.1"` |
| `pyz3/pyZ3-template/cookiecutter.json` | Script | `"version": "0.1.1"` |
| `poetry.lock` | Poetry | Package metadata |

---

## Troubleshooting

### Version Mismatch Between Files

If `pyproject.toml` and `pyz3/__init__.py` have different versions:

```bash
# Get version from pyproject.toml
VERSION=$(poetry version -s)

# Update __init__.py to match
sed -i '' "s/^__version__ = .*/__version__ = \"$VERSION\"/" pyz3/__init__.py

# Update cookiecutter template to match
sed -i '' "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" pyz3/pyZ3-template/cookiecutter.json
```

### Script Not Executable

```bash
chmod +x bump_version.sh
```

### Poetry Not Found

Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

---

## Best Practices

1. ✅ **Always use bump rules** (patch, minor, major) instead of explicit versions when possible
2. ✅ **Review changes** with `git diff` before committing
3. ✅ **Use dry-run** to preview version bumps: `./bump_version.sh patch --dry-run`
4. ✅ **Tag releases** with `v` prefix (e.g., `v0.1.1`)
5. ✅ **Write release notes** describing what changed
6. ✅ **Test before releasing** - run `make test-all`

---

## Quick Commands Reference

```bash
# Show version
make version
poetry version

# Bump patch (0.1.0 → 0.1.1)
make bump-patch
./bump_version.sh patch
poetry version patch

# Bump minor (0.1.0 → 0.2.0)
make bump-minor
./bump_version.sh minor
poetry version minor

# Bump major (0.1.0 → 1.0.0)
make bump-major
./bump_version.sh major
poetry version major

# Set specific version
./bump_version.sh 0.2.0-beta.1

# Dry run
./bump_version.sh patch --dry-run
```

---

## See Also

- [Poetry Version Documentation](https://python-poetry.org/docs/cli/#version)
- [Semantic Versioning Specification](https://semver.org/)
- [GitHub Releases Guide](https://docs.github.com/en/repositories/releasing-projects-on-github)
