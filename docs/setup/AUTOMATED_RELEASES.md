# Automated Releases

**Status**: ✅ Configured
**Trigger**: Push to `main` branch
**No manual tagging required!**

## How It Works

Every time you push to the `main` branch, the workflow automatically:

1. ✅ **Builds wheels** for all platforms (12 wheels total)
2. ✅ **Reads version** from `pyproject.toml`
3. ✅ **Creates git tag** (e.g., `v0.1.0`) if it doesn't exist
4. ✅ **Creates GitHub Release** with all wheels attached
5. ✅ **Publishes to PyPI** (if configured)

## Quick Start

### 1. Update Version

Edit `pyproject.toml`:

```toml
[tool.poetry]
version = "0.1.0"  # Change this to your desired version
```

And `pyz3/__init__.py`:

```python
__version__ = "0.1.0"  # Keep in sync with pyproject.toml
```

### 2. Commit and Push

```bash
# Make your changes
git add .
git commit -m "Your changes"

# Push to main - this triggers everything!
git push origin main
```

That's it! No manual tagging needed.

## What Happens Next

### Automatic Process

```
Push to main
    ↓
Build 12 wheels (Linux, macOS, Windows × Python 3.11, 3.12, 3.13)
    ↓
Extract version from pyproject.toml (e.g., "0.1.0")
    ↓
Check if tag "v0.1.0" exists
    ↓
If NOT exists:
    ├─ Create tag "v0.1.0"
    ├─ Push tag to GitHub
    └─ Create GitHub Release with wheels
    ↓
Publish to PyPI (skip if already exists)
```

### Timeline

- **Build wheels**: ~5-10 minutes
- **Create release**: ~1 minute
- **Publish to PyPI**: ~1 minute

**Total**: ~7-12 minutes from push to published release

## Version Bump Workflow

When you want to release a new version:

### Patch Release (0.1.0 → 0.1.1)

```bash
# 1. Update version
sed -i '' 's/version = "0.1.0"/version = "0.1.1"/' pyproject.toml
sed -i '' 's/__version__ = "0.1.0"/__version__ = "0.1.1"/' pyz3/__init__.py

# 2. Commit
git add pyproject.toml pyz3/__init__.py
git commit -m "Bump version to 0.1.1"

# 3. Push - automatic release!
git push origin main
```

### Minor Release (0.1.0 → 0.2.0)

```bash
# 1. Update version
sed -i '' 's/version = "0.1.0"/version = "0.2.0"/' pyproject.toml
sed -i '' 's/__version__ = "0.1.0"/__version__ = "0.2.0"/' pyz3/__init__.py

# 2. Commit
git add pyproject.toml pyz3/__init__.py
git commit -m "Release v0.2.0: New features

- Feature 1
- Feature 2
"

# 3. Push
git push origin main
```

### Major Release (0.1.0 → 1.0.0)

```bash
# 1. Update version
sed -i '' 's/version = "0.1.0"/version = "1.0.0"/' pyproject.toml
sed -i '' 's/__version__ = "0.1.0"/__version__ = "1.0.0"/' pyz3/__init__.py

# 2. Commit
git add pyproject.toml pyz3/__init__.py
git commit -m "Release v1.0.0: Major release

Breaking changes:
- Change 1
- Change 2

New features:
- Feature 1
- Feature 2
"

# 3. Push
git push origin main
```

## Version Synchronization

**Important**: Keep these two files in sync!

### File 1: `pyproject.toml`
```toml
[tool.poetry]
version = "0.1.0"  # ← Source of truth
```

### File 2: `pyz3/__init__.py`
```python
__version__ = "0.1.0"  # ← Must match
```

### Helper Script

Create a script to update both at once:

```bash
#!/bin/bash
# bump_version.sh

if [ -z "$1" ]; then
  echo "Usage: ./bump_version.sh <new_version>"
  echo "Example: ./bump_version.sh 0.1.1"
  exit 1
fi

NEW_VERSION=$1

# Update pyproject.toml
sed -i '' "s/^version = .*/version = \"$NEW_VERSION\"/" pyproject.toml

# Update __init__.py
sed -i '' "s/^__version__ = .*/__version__ = \"$NEW_VERSION\"/" pyz3/__init__.py

echo "✅ Updated version to $NEW_VERSION"
echo "Next steps:"
echo "  git add pyproject.toml pyz3/__init__.py"
echo "  git commit -m 'Bump version to $NEW_VERSION'"
echo "  git push origin main"
```

Usage:
```bash
chmod +x bump_version.sh
./bump_version.sh 0.1.1
```

## Workflow Conditions

### When Release Happens

✅ **Creates release** when:
- Push to `main` branch
- Tag doesn't already exist
- All builds succeed

❌ **Skips release** when:
- Push to `develop` or other branches
- Tag already exists (prevents duplicates)
- Pull request (builds but doesn't release)

### Manual Override

If you need to trigger manually:

1. Go to GitHub Actions
2. Select "Build pyZ3 Wheels" workflow
3. Click "Run workflow"
4. Select branch (main)
5. Click "Run workflow"

## PyPI Publishing

### First Time Setup

1. **Create PyPI account**: https://pypi.org/account/register/

2. **Configure trusted publishing**:
   - Go to https://pypi.org/manage/account/publishing/
   - Add: `yourusername/pyZ3` repository
   - Workflow: `build-wheels.yml`
   - Environment: `pypi`

3. **Test with TestPyPI first** (optional):
   ```yaml
   # Temporarily change in workflow
   environment:
     name: testpypi
     url: https://test.pypi.org/p/pyZ3
   ```

### Disable PyPI Publishing

If you don't want to publish to PyPI yet:

```yaml
# Comment out the publish job in .github/workflows/build-wheels.yml
# publish:
#   name: Publish to PyPI
#   ...
```

## Monitoring Releases

### Check Release Status

```bash
# View all releases
gh release list

# View latest release
gh release view

# Download release assets
gh release download v0.1.0
```

### Check PyPI

```bash
# View package on PyPI
pip index versions pyZ3

# Install specific version
pip install pyZ3==0.1.0
```

## Troubleshooting

### Tag Already Exists

**Error**: `tag 'v0.1.0' already exists`

**Solution**: The workflow checks and skips if tag exists. Bump version in `pyproject.toml`.

### PyPI Publishing Fails

**Error**: `Package already exists`

**Solution**: Workflow uses `skip-existing: true`, so this is fine. Already published versions are skipped.

### Build Fails

**Error**: Any build failure

**Solution**: Release won't be created. Fix the error and push again.

## Benefits

### For Developers
- ✅ No manual tagging
- ✅ No manual release creation
- ✅ No manual PyPI publishing
- ✅ Just bump version and push

### For Users
- ✅ Consistent release cadence
- ✅ All platforms supported
- ✅ Automatic changelog from commits
- ✅ Easy to install: `pip install pyZ3`

## Best Practices

### 1. Use Conventional Commits

```bash
git commit -m "feat: Add NumPy integration"
git commit -m "fix: Resolve memory leak"
git commit -m "docs: Update README"
```

### 2. Test Before Pushing

```bash
# Run tests locally
pytest

# Build locally
zig build

# Install locally
pip install -e .
```

### 3. Use Develop Branch

```bash
# Work on develop
git checkout develop
git commit -m "Work in progress"
git push origin develop  # No release

# When ready, merge to main
git checkout main
git merge develop
git push origin main  # Triggers release!
```

### 4. Version Strategy

- **Patch** (0.1.0 → 0.1.1): Bug fixes
- **Minor** (0.1.0 → 0.2.0): New features, backward compatible
- **Major** (0.1.0 → 1.0.0): Breaking changes

## Advanced: Pre-releases

For pre-release versions:

```toml
# pyproject.toml
version = "0.2.0-beta.1"
```

The workflow will create tag `v0.2.0-beta.1` and mark as pre-release.

## Summary

```bash
# The entire release process:
1. Edit version in pyproject.toml and pyz3/__init__.py
2. git commit -m "Release vX.Y.Z"
3. git push origin main

# That's it! Everything else is automatic:
# - Builds 12 wheels
# - Creates git tag
# - Creates GitHub release
# - Publishes to PyPI
```

**No manual steps required! 🎉**

---

**Configured**: 2025-12-06
**Workflow**: `.github/workflows/build-wheels.yml`
**Status**: ✅ Ready to use
