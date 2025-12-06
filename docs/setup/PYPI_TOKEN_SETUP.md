# PyPI Token Setup

## Current Setup: Trusted Publishing (Recommended)

The workflow is configured to use **OpenID Connect (OIDC)** trusted publishing, which is more secure and doesn't require managing tokens.

**No token needed!** Just configure on PyPI.

### Setup Trusted Publishing

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - **PyPI Project Name**: `pyZ3`
   - **Owner**: `YOUR_GITHUB_USERNAME`
   - **Repository name**: `pyZ3`
   - **Workflow name**: `build-wheels.yml`
   - **Environment name**: `pypi`
4. Click "Add"

That's it! No secrets needed.

---

## Alternative: Using PyPI Token

If you prefer using a token instead:

### Environment Variable Name

**Variable**: `PYPI_API_TOKEN`

### Step 1: Create PyPI Token

1. Create account: https://pypi.org/account/register/
2. Go to: https://pypi.org/manage/account/token/
3. Click "Add API token"
4. Settings:
   - **Token name**: `pyZ3-github-actions`
   - **Scope**: Project: `pyZ3` (or "Entire account" if project doesn't exist yet)
5. Click "Add token"
6. **IMPORTANT**: Copy the token immediately (starts with `pypi-`)
   - Example: `pypi-AgEIcHlwaS5vcmcCJGFiY2RlZi0xMjM0LTU2NzgtOTBhYi1jZGVmZ2hpamtsbW4...`

### Step 2: Add to GitHub Secrets

1. Go to: `https://github.com/YOUR_USERNAME/pyZ3/settings/secrets/actions`
2. Click "New repository secret"
3. Enter:
   - **Name**: `PYPI_API_TOKEN`
   - **Secret**: Paste the token (starts with `pypi-`)
4. Click "Add secret"

### Step 3: Update Workflow

Edit `.github/workflows/build-wheels.yml`:

```yaml
# Change from trusted publishing:
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    packages-dir: dist/
    skip-existing: true
  # No password needed with trusted publishing

# To token-based:
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    packages-dir: dist/
    skip-existing: true
    password: ${{ secrets.PYPI_API_TOKEN }}  # ← Add this line
```

Also remove the `id-token: write` permission:

```yaml
publish:
  name: Publish to PyPI
  needs: auto_release
  runs-on: ubuntu-latest
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  environment:
    name: pypi
    url: https://pypi.org/p/pyz3
  permissions:
    # Remove this line if using token:
    # id-token: write
    contents: read  # Add this instead
```

---

## Comparison

| Feature | Trusted Publishing | Token |
|---------|-------------------|-------|
| **Security** | ✅ More secure (no secrets) | ⚠️ Token can leak |
| **Setup** | Easy (one-time on PyPI) | Requires GitHub secrets |
| **Rotation** | No rotation needed | Should rotate periodically |
| **Revocation** | Automatic if workflow changes | Manual revocation |
| **Recommended** | ✅ Yes | For legacy/specific needs |

---

## TestPyPI Token (For Testing)

To test with TestPyPI first:

### Step 1: Create TestPyPI Token

1. Create account: https://test.pypi.org/account/register/
2. Go to: https://test.pypi.org/manage/account/token/
3. Create token named: `pyZ3-test`
4. Copy token

### Step 2: Add GitHub Secret

- **Name**: `TEST_PYPI_API_TOKEN`
- **Value**: Paste the token

### Step 3: Create Test Workflow

Or modify the workflow to publish to TestPyPI first:

```yaml
- name: Publish to TestPyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    repository-url: https://test.pypi.org/legacy/
    packages-dir: dist/
    skip-existing: true
    password: ${{ secrets.TEST_PYPI_API_TOKEN }}
```

---

## Quick Reference

### Environment Variables

| Variable | Purpose | Where to Set |
|----------|---------|--------------|
| `PYPI_API_TOKEN` | Production PyPI | GitHub Secrets |
| `TEST_PYPI_API_TOKEN` | TestPyPI testing | GitHub Secrets |

### Workflow Syntax

```yaml
# Using token
password: ${{ secrets.PYPI_API_TOKEN }}

# Using trusted publishing (current setup)
# No password line needed, requires:
permissions:
  id-token: write
```

---

## Recommended Setup (Current)

**Keep trusted publishing!** It's more secure:

1. ✅ No secrets to manage
2. ✅ No token rotation needed
3. ✅ Automatic security updates
4. ✅ GitHub/PyPI handle authentication

**Setup once on PyPI**: https://pypi.org/manage/account/publishing/

---

## Troubleshooting

### Error: "Invalid or expired API token"

**Solution**:
1. Regenerate token on PyPI
2. Update `PYPI_API_TOKEN` secret in GitHub
3. Re-run workflow

### Error: "Trusted publisher not found"

**Solution**:
1. Go to https://pypi.org/manage/account/publishing/
2. Add pending publisher with exact values:
   - Repository: `YOUR_USERNAME/pyZ3`
   - Workflow: `build-wheels.yml`
   - Environment: `pypi`

### Error: "Package already exists"

**Solution**:
Workflow uses `skip-existing: true`, so this should be automatic.
If you see this error, bump version number.

---

## Summary

**Current Setup** (No changes needed):
- ✅ Uses trusted publishing (OIDC)
- ✅ No token required
- ✅ More secure
- ✅ No secrets to manage

**If you want to use a token**:
- Variable name: `PYPI_API_TOKEN`
- Add to: GitHub Secrets
- Update workflow with: `password: ${{ secrets.PYPI_API_TOKEN }}`

**Recommendation**: Keep current trusted publishing setup!

---

**Date**: 2025-12-06
**Current**: Trusted Publishing (OIDC)
**Alternative**: Token-based (documented above)
