#!/bin/bash
# bump_version.sh - Update version in pyz3

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_usage() {
  echo -e "${BLUE}Usage:${NC} $0 <version>"
  echo ""
  echo -e "${YELLOW}Examples:${NC}"
  echo "  $0 0.9.2        # Set version to 0.9.2"
  echo "  $0 1.0.0        # Set version to 1.0.0"
  echo "  $0 1.0.0-beta.1 # Set version to 1.0.0-beta.1"
  echo ""
}

if [ -z "$1" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  show_usage
  exit 0
fi

NEW_VERSION=$1

# Validate version format (simple check)
if ! echo "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-z0-9\.]+)?$'; then
  echo -e "${RED}Error: Invalid version format: $NEW_VERSION${NC}"
  echo "Expected format: X.Y.Z or X.Y.Z-suffix"
  exit 1
fi

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep -E '^version = "' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
echo -e "${YELLOW}Current version:${NC} $CURRENT_VERSION"
echo -e "${YELLOW}New version:${NC}     $NEW_VERSION"
echo ""

# Update pyproject.toml
echo -n "Updating pyproject.toml... "
if sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml; then
  echo -e "${GREEN}OK${NC}"
else
  echo -e "${RED}FAIL${NC}"
  exit 1
fi

# Update __init__.py
echo -n "Updating pyz3/__init__.py... "
if sed -i '' "s/^__version__ = .*/__version__ = \"$NEW_VERSION\"/" pyz3/__init__.py; then
  echo -e "${GREEN}OK${NC}"
else
  echo -e "${RED}FAIL${NC}"
  exit 1
fi

# Update pyZ3-template files
TEMPLATE_DIR="pyz3/pyZ3-template"
TEMPLATE_PROJECT_DIR="$TEMPLATE_DIR/{{cookiecutter.project_slug}}"

# Update template pyproject.toml (pyz3 dependency versions)
TEMPLATE_PYPROJECT="$TEMPLATE_PROJECT_DIR/pyproject.toml"
if [ -f "$TEMPLATE_PYPROJECT" ]; then
  echo -n "Updating template pyproject.toml... "
  # Update build-system requires: pyz3>=X.Y.Z
  if sed -i '' "s/pyz3>=[0-9.]*\"/pyz3>=$NEW_VERSION\"/" "$TEMPLATE_PYPROJECT" && \
     # Update dev dependencies: pyz3 = ">=X.Y.Z"
     sed -i '' "s/pyz3 = \">=[0-9.]*\"/pyz3 = \">=$NEW_VERSION\"/" "$TEMPLATE_PYPROJECT"; then
    echo -e "${GREEN}OK${NC}"
  else
    echo -e "${RED}FAIL${NC}"
    exit 1
  fi
fi

echo ""
echo -e "${GREEN}[DONE] Version successfully updated to $NEW_VERSION${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Review changes: git diff"
echo "  2. Commit changes:"
echo "     git add pyproject.toml pyz3/__init__.py pyz3/pyZ3-template/"
echo "     git commit -m \"Bump version to $NEW_VERSION\""
echo "  3. Create git tag:"
echo "     git tag v$NEW_VERSION"
echo "  4. Push changes:"
echo "     git push origin main --tags"
echo ""
