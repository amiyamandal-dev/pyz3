#!/bin/bash
# bump_version.sh - Automatically update version in both files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ -z "$1" ]; then
  echo -e "${RED}Error: Version argument required${NC}"
  echo ""
  echo "Usage: $0 <new_version>"
  echo ""
  echo "Examples:"
  echo "  $0 0.1.1        # Patch release"
  echo "  $0 0.2.0        # Minor release"
  echo "  $0 1.0.0        # Major release"
  echo "  $0 0.2.0-beta.1 # Pre-release"
  exit 1
fi

NEW_VERSION=$1

# Validate version format (basic check)
if ! [[ $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-z0-9\.]+)?$ ]]; then
  echo -e "${RED}Error: Invalid version format${NC}"
  echo "Expected format: X.Y.Z or X.Y.Z-pre"
  echo "Examples: 0.1.0, 1.0.0, 0.2.0-beta.1"
  exit 1
fi

# Get current version
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

echo -e "${YELLOW}Current version:${NC} $CURRENT_VERSION"
echo -e "${YELLOW}New version:${NC}     $NEW_VERSION"
echo ""

# Confirm
read -p "Update version? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Cancelled"
  exit 0
fi

# Update pyproject.toml
echo -n "Updating pyproject.toml... "
if sed -i '' "s/^version = .*/version = \"$NEW_VERSION\"/" pyproject.toml; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${RED}✗${NC}"
  exit 1
fi

# Update __init__.py
echo -n "Updating pyz3/__init__.py... "
if sed -i '' "s/^__version__ = .*/__version__ = \"$NEW_VERSION\"/" pyz3/__init__.py; then
  echo -e "${GREEN}✓${NC}"
else
  echo -e "${RED}✗${NC}"
  exit 1
fi

echo ""
echo -e "${GREEN}✅ Version successfully updated to $NEW_VERSION${NC}"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Commit: git add pyproject.toml pyz3/__init__.py"
echo "  3.         git commit -m 'Bump version to $NEW_VERSION'"
echo "  4. Push:   git push origin main"
echo ""
echo "After pushing to main:"
echo "  - GitHub Actions will build 12 wheels"
echo "  - Create git tag v$NEW_VERSION"
echo "  - Create GitHub release"
echo "  - Publish to PyPI (if configured)"
echo ""
echo "ETA: ~10 minutes"
