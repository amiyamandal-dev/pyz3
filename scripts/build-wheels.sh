#!/bin/bash
# Convenience script for building wheels

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}pyz3 Wheel Builder${NC}\n"

# Parse arguments
PLATFORM=""
ALL_PLATFORMS=false
OPTIMIZE="ReleaseFast"
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --all-platforms)
            ALL_PLATFORMS=true
            shift
            ;;
        --optimize)
            OPTIMIZE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --platform PLATFORM    Build for specific platform"
            echo "                         (linux-x86_64, linux-aarch64, macos-x86_64,"
            echo "                          macos-arm64, windows-x64)"
            echo "  --all-platforms        Build for all platforms"
            echo "  --optimize LEVEL       Optimization level (Debug, ReleaseSafe,"
            echo "                         ReleaseFast, ReleaseSmall)"
            echo "  -v, --verbose          Verbose output"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Build arguments
BUILD_ARGS=()

if [ "$ALL_PLATFORMS" = true ]; then
    BUILD_ARGS+=(--all-platforms)
elif [ -n "$PLATFORM" ]; then
    BUILD_ARGS+=(--platform "$PLATFORM")
fi

BUILD_ARGS+=(--optimize "$OPTIMIZE")

if [ -n "$VERBOSE" ]; then
    BUILD_ARGS+=($VERBOSE)
fi

# Run the wheel builder
echo -e "${BLUE}Building wheels...${NC}\n"

python -m pyz3.wheel "${BUILD_ARGS[@]}"

echo -e "\n${GREEN}✓ Done!${NC}"
echo -e "\nWheels saved to: ${BLUE}dist/${NC}"
ls -lh dist/*.whl 2>/dev/null || echo "No wheels found"
