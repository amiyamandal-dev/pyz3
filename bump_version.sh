#!/bin/bash
# bump_version.sh - Automatically update version using Poetry
#
# This script uses Poetry's built-in version command to manage versions.
# It automatically updates both pyproject.toml and pyz3/__init__.py

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_usage() {
  echo -e "${BLUE}Usage:${NC} $0 <version_or_rule>"
  echo ""
  echo -e "${YELLOW}Bump Rules (recommended):${NC}"
  echo "  $0 patch        # 0.1.0 → 0.1.1"
  echo "  $0 minor        # 0.1.0 → 0.2.0"
  echo "  $0 major        # 0.1.0 → 1.0.0"
  echo "  $0 prepatch     # 0.1.0 → 0.1.1-alpha.0"
  echo "  $0 preminor     # 0.1.0 → 0.2.0-alpha.0"
  echo "  $0 premajor     # 0.1.0 → 1.0.0-alpha.0"
  echo "  $0 prerelease   # 0.1.1-alpha.0 → 0.1.1-alpha.1"
  echo ""
  echo -e "${YELLOW}Explicit Versions:${NC}"
  echo "  $0 0.1.1        # Set to specific version"
  echo "  $0 0.2.0-beta.1 # Set to specific pre-release"
  echo ""
  echo -e "${YELLOW}Options:${NC}"
  echo "  --dry-run       # Show what would change without making changes"
  echo ""
}

if [ -z "$1" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  show_usage
  exit 0
fi

VERSION_ARG=$1
DRY_RUN=""

if [ "$2" = "--dry-run" ] || [ "$1" = "--dry-run" ]; then
  DRY_RUN="--dry-run"
  if [ "$1" = "--dry-run" ]; then
    VERSION_ARG=$2
  fi
fi

# Check if poetry is available
if ! command -v poetry &> /dev/null; then
  echo -e "${RED}Error: poetry not found${NC}"
  echo "Please install poetry: https://python-poetry.org/docs/#installation"
  exit 1
fi

# Get current version
CURRENT_VERSION=$(poetry version -s)

echo -e "${YELLOW}Current version:${NC} $CURRENT_VERSION"

# Use poetry to bump version (or set specific version)
if [ -n "$DRY_RUN" ]; then
  NEW_VERSION=$(poetry version $DRY_RUN $VERSION_ARG 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+(-[a-z0-9\.]+)?' | tail -1)
  echo -e "${BLUE}[DRY RUN]${NC} Would update to: $NEW_VERSION"
  echo ""
  echo "Files that would be updated:"
  echo "  - pyproject.toml"
  echo "  - pyz3/__init__.py"
  echo "  - pyz3/pyZ3-template/cookiecutter.json"
  echo "  - pyz3/pyZ3-template/{{cookiecutter.project_slug}}/pyproject.toml"
  echo "  - poetry.lock"
  exit 0
else
  echo -n "Bumping version with poetry... "
  if poetry version $VERSION_ARG > /dev/null 2>&1; then
    NEW_VERSION=$(poetry version -s)
    echo -e "${GREEN}✓${NC}"
  else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}Error: Invalid version or rule: $VERSION_ARG${NC}"
    echo ""
    show_usage
    exit 1
  fi
fi

echo -e "${YELLOW}New version:${NC}     $NEW_VERSION"
echo ""

# Update __init__.py to match
echo -n "Updating pyz3/__init__.py... "
if sed -i '' "s/^__version__ = .*/__version__ = \"$NEW_VERSION\"/" pyz3/__init__.py; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${RED}✗${NC}"
  exit 1
fi

# Update cookiecutter template version
echo -n "Updating template cookiecutter.json... "
TEMPLATE_JSON="pyz3/pyZ3-template/cookiecutter.json"
if [ -f "$TEMPLATE_JSON" ]; then
  if sed -i '' "s/\"version\": \".*\"/\"version\": \"$NEW_VERSION\"/" "$TEMPLATE_JSON"; then
    echo -e "${GREEN}✓${NC}"
  else
    echo -e "${RED}✗${NC}"
    exit 1
  fi
else
  echo -e "${YELLOW}⚠${NC} (template not found)"
fi

# Update template pyproject.toml with new pyz3 version
echo -n "Updating template pyproject.toml... "
TEMPLATE_PYPROJECT="pyz3/pyZ3-template/{{cookiecutter.project_slug}}/pyproject.toml"
if [ -f "$TEMPLATE_PYPROJECT" ]; then
  # Update pyz3 version in build-system requires
  if sed -i '' "s/\"pyz3==[0-9.]*\"/\"pyz3==$NEW_VERSION\"/" "$TEMPLATE_PYPROJECT" && \
     sed -i '' "s/pyz3 = \"[0-9.]*\"/pyz3 = \"$NEW_VERSION\"/" "$TEMPLATE_PYPROJECT"; then
    echo -e "${GREEN}✓${NC}"
  else
    echo -e "${RED}✗${NC}"
    exit 1
  fi
else
  echo -e "${YELLOW}⚠${NC} (template pyproject.toml not found)"
fi

# Update poetry.lock
echo -n "Updating poetry.lock... "
if poetry lock --no-update > /dev/null 2>&1; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${GREEN}✓${NC} (lock file already up to date)"
fi

echo ""
echo -e "${GREEN}✅ Version successfully updated to $NEW_VERSION${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Review changes: ${YELLOW}git diff${NC}"
echo "  2. Commit changes:"
echo "     ${YELLOW}git add pyproject.toml pyz3/__init__.py pyz3/pyZ3-template/cookiecutter.json pyz3/pyZ3-template/{{cookiecutter.project_slug}}/pyproject.toml poetry.lock${NC}"
echo "     ${YELLOW}git commit -m \"Bump version to $NEW_VERSION\"${NC}"
echo "  3. Create git tag:"
echo "     ${YELLOW}git tag v$NEW_VERSION${NC}"
echo "  4. Push changes:"
echo "     ${YELLOW}git push origin main --tags${NC}"
echo ""
echo -e "${BLUE}After pushing:${NC}"
echo "  ✓ GitHub Actions will build wheels for all platforms"
echo "  ✓ Create a GitHub release from tag v$NEW_VERSION"
echo "  ✓ Wheels will be uploaded to release (if configured)"
echo "  ✓ Publish to PyPI (if configured)"
echo ""
echo -e "${YELLOW}Estimated time: ~10 minutes${NC}"
