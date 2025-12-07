# pyZ3 Setup Checklist

Use this checklist to complete the setup of your pyZ3 fork.

## ✅ Completed (Automated Rename)

- [x] Rename package from `pydust` to `pyz3`
- [x] Update all Zig imports
- [x] Rename directory structure
- [x] Update pyproject.toml
- [x] Update GitHub workflows
- [x] Update all documentation
- [x] Update template files
- [x] Create fork documentation

## 📝 Manual Steps Required

### 1. Update Personal Information

Edit `pyproject.toml`:
```toml
[tool.poetry]
name = "pyZ3"
authors = ["Your Name <your.email@example.com>"]  # ← Update this
homepage = "https://github.com/amiyamandal-dev/pyz3"  # ← Update this
repository = "https://github.com/amiyamandal-dev/pyz3"  # ← Update this
```

### 2. Update README.md URLs

Replace all instances of `yourusername` with your actual GitHub username:
```bash
sed -i '' 's/yourusername/YOUR_GITHUB_USERNAME/g' README.md
sed -i '' 's/yourusername/YOUR_GITHUB_USERNAME/g' mkdocs.yml
```

### 3. Create GitHub Repository

1. Go to GitHub and create a new repository named `pyZ3`
2. Make it public (recommended) or private
3. Do NOT initialize with README (we already have one)

### 4. Update Git Remote

```bash
# Remove old remote (if exists)
git remote remove origin

# Add new remote
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/pyZ3.git

# Verify
git remote -v
```

### 5. Initial Commit and Push

```bash
# Stage all changes
git add -A

# Commit
git commit -m "Initial commit: Fork ziggy-pydust as pyZ3

Major changes:
- Renamed package from pydust to pyz3
- Added NumPy integration with zero-copy array access
- Updated all imports and references
- Enhanced cross-compilation support
- Comprehensive NumPy documentation

Forked from: https://github.com/fulcrum-so/ziggy-pydust
License: Apache 2.0 (maintained)"

# Push to main branch
git branch -M main
git push -u origin main
```

### 6. Set Up PyPI (Optional)

If you want to publish to PyPI:

1. **Create PyPI account** (if you don't have one)
   - Go to https://pypi.org/account/register/

2. **Create API token**
   - Go to https://pypi.org/manage/account/token/
   - Create a token for the pyZ3 project

3. **Configure GitHub secrets**
   - Go to your repository → Settings → Secrets and variables → Actions
   - Add `PYPI_API_TOKEN` secret with your token

4. **Enable trusted publishing** (recommended)
   - Configure at https://pypi.org/manage/account/publishing/

### 7. Test the Setup

```bash
# 1. Install in development mode
pip install -e .

# 2. Test import
python -c "import pyz3; print('✅ Import works')"

# 3. Test CLI
pyz3 --help

# 4. Run build
zig build

# 5. Run tests
pytest

# 6. Test template
cd /tmp
pyz3 init -n testproject --description "Test" --email "test@test.com" --no-interactive
cd testproject
zig build
pytest
```

### 8. Update Documentation (Optional)

If you want to customize the documentation:

1. **Edit docs/index.md**
   - Add your project introduction
   - Update getting started guide

2. **Build docs locally**
   ```bash
   poetry install --with docs
   poetry run mkdocs serve
   # Visit http://localhost:8000
   ```

3. **Deploy docs** (GitHub Pages)
   ```bash
   poetry run mike deploy --push --update-aliases 0.1 latest
   ```

### 9. Create First Release

When you're ready for your first release:

```bash
# 1. Tag the release
git tag -a v0.1.0 -m "Release v0.1.0: Initial pyZ3 fork with NumPy integration"

# 2. Push tag
git push origin v0.1.0

# This will trigger:
# - GitHub Actions to build wheels for all platforms
# - Automatic PyPI publishing (if configured)
# - GitHub release creation
```

### 10. Announce Your Fork

Consider announcing on:
- Reddit (r/Python, r/Zig)
- Hacker News
- Your blog or social media
- Zig Discord/Forum

Example announcement:
```
Introducing pyZ3 - A Python extension framework in Zig

I've forked ziggy-pydust to create pyZ3, adding:
✅ Built-in NumPy integration with zero-copy array access
✅ Enhanced cross-compilation for multi-platform wheels
✅ Comprehensive NumPy documentation and examples

Perfect for building high-performance Python extensions for data science!

GitHub: https://github.com/YOUR_USERNAME/pyZ3
PyPI: https://pypi.org/project/pyZ3/

Thanks to the original ziggy-pydust team for the excellent foundation!
```

## 🔍 Verification Commands

Run these to ensure everything is working:

```bash
# Check for remaining old references
grep -r "pydust" --exclude-dir=.git --exclude-dir=.venv --exclude="*.md" .
grep -r "ziggy-pydust" --exclude-dir=.git --exclude-dir=.venv --exclude="*.md" .

# Should return no results (except in documentation/attribution files)
```

## 📋 Post-Setup Tasks

- [ ] Add repository description on GitHub
- [ ] Add topics/tags: `python`, `zig`, `numpy`, `performance`, `extensions`
- [ ] Enable GitHub Issues
- [ ] Enable GitHub Discussions (optional)
- [ ] Add repository banner/logo (optional)
- [ ] Update social media links in mkdocs.yml
- [ ] Create CHANGELOG.md for tracking releases
- [ ] Set up branch protection rules (optional)

## 🎯 Development Workflow

Your ongoing workflow should be:

```bash
# 1. Make changes
vim pyz3/src/types/new_feature.zig

# 2. Test locally
zig build test
pytest

# 3. Commit
git add .
git commit -m "Add new feature: ..."

# 4. Push
git push origin main

# 5. When ready for release
git tag v0.2.0
git push origin v0.2.0  # Triggers CI/CD
```

## 📚 Resources

- **Original Project**: https://github.com/fulcrum-so/ziggy-pydust
- **Zig Documentation**: https://ziglang.org/documentation/
- **Python Packaging**: https://packaging.python.org/
- **NumPy C API**: https://numpy.org/doc/stable/reference/c-api/

## 🆘 Need Help?

If you run into issues:
1. Check RENAME_SUMMARY.md for migration guidance
2. Review FORK_NOTICE.md for architectural changes
3. Look at example files for usage patterns
4. Open an issue on your repository

---

**Last Updated**: 2025-12-06
**Status**: Ready for independent development
**Next Step**: Update personal information in pyproject.toml
