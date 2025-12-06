# Quick Release Guide

## TL;DR - How to Release

```bash
# 1. Update version
./bump_version.sh 0.1.1  # or edit manually

# 2. Commit and push
git add pyproject.toml pyz3/__init__.py
git commit -m "Release v0.1.1"
git push origin main

# 3. Wait ~10 minutes
# ✅ Wheels built
# ✅ GitHub release created
# ✅ Published to PyPI
```

**That's it! No manual tagging needed.**

## First Time Only

### 1. Create bump_version.sh

```bash
cat > bump_version.sh << 'EOF'
#!/bin/bash
NEW_VERSION=$1
sed -i '' "s/^version = .*/version = \"$NEW_VERSION\"/" pyproject.toml
sed -i '' "s/^__version__ = .*/__version__ = \"$NEW_VERSION\"/" pyz3/__init__.py
echo "✅ Version updated to $NEW_VERSION"
EOF

chmod +x bump_version.sh
```

### 2. Update Your Info

Edit `pyproject.toml`:
```toml
authors = ["Your Name <your.email@example.com>"]
homepage = "https://github.com/YOUR_USERNAME/pyZ3"
repository = "https://github.com/YOUR_USERNAME/pyZ3"
```

Edit `pyz3/__init__.py`:
```python
__author__ = "Your Name"
__email__ = "your.email@example.com"
```

### 3. Setup PyPI (Optional)

1. Create account: https://pypi.org/account/register/
2. Configure trusted publishing: https://pypi.org/manage/account/publishing/
   - Repository: `YOUR_USERNAME/pyZ3`
   - Workflow: `build-wheels.yml`
   - Environment: `pypi`

## Release Checklist

- [ ] All tests passing locally (`pytest`)
- [ ] Version bumped in both files
- [ ] Commit message describes changes
- [ ] Pushed to `main` branch
- [ ] Wait for GitHub Actions (check Actions tab)
- [ ] Verify release created (Releases tab)
- [ ] Verify PyPI upload (if configured)

## Common Versions

```bash
# Bug fix
./bump_version.sh 0.1.1

# New feature
./bump_version.sh 0.2.0

# Breaking change
./bump_version.sh 1.0.0

# Pre-release
./bump_version.sh 0.2.0-beta.1
```

## Troubleshooting

**Builds failing?**
- Check GitHub Actions logs
- Verify all fixes from BUILD_FIXES_SUMMARY.md are applied

**PyPI publishing fails?**
- Check if version already exists (workflow skips automatically)
- Verify trusted publishing is configured

**Tag already exists?**
- Workflow checks and skips
- Bump to a new version number

## What Gets Released

Every push to `main` creates:

- ✅ 12 platform wheels:
  - Linux x86_64 (Python 3.11, 3.12, 3.13)
  - macOS x86_64 (Python 3.11, 3.12, 3.13)
  - macOS arm64 (Python 3.11, 3.12, 3.13)
  - Windows x64 (Python 3.11, 3.12, 3.13)

- ✅ Git tag (e.g., `v0.1.0`)
- ✅ GitHub Release with all wheels
- ✅ PyPI package (if configured)

## Next Steps

1. Read AUTOMATED_RELEASES.md for full details
2. Test with a version bump: `./bump_version.sh 0.1.0`
3. Push and watch the magic happen!

---

**Quick Start**: `./bump_version.sh 0.1.0 && git add . && git commit -m "Release v0.1.0" && git push origin main`
