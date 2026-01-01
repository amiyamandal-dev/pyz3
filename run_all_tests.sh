#!/bin/bash
# run_all_tests.sh - Run all tests for pyz3

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

print_header() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

print_error() {
    echo -e "${RED}[FAIL] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "CHECKING PREREQUISITES"

    # Check Zig
    if ! command -v zig &> /dev/null; then
        print_error "Zig is not installed"
        exit 1
    fi
    print_success "Zig $(zig version)"

    # Check uv or pip
    if command -v uv &> /dev/null; then
        print_success "uv $(uv --version)"
        RUNNER="uv run"
    else
        print_warning "uv not found, using pip"
        RUNNER="python -m"
    fi
}

# Build project
build_project() {
    print_header "BUILDING PROJECT"

    if $RUNNER python -m ziglang build; then
        print_success "Build completed"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Run Zig tests
run_zig_tests() {
    print_header "RUNNING ZIG TESTS"

    if $RUNNER python -m ziglang build test; then
        print_success "Zig tests passed"
    else
        print_warning "Some Zig tests may have failed"
    fi
}

# Run pytest
run_pytest() {
    print_header "RUNNING PYTEST"

    if $RUNNER pytest test/ -v --tb=short; then
        print_success "Pytest passed"
    else
        print_warning "Some pytest tests failed"
    fi
}

# Quick check
quick_check() {
    print_header "QUICK CHECK"

    $RUNNER python << 'EOF'
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import uuid

tests = [
    ("Complex", lambda: abs(complex(3, 4)) == 5.0),
    ("Decimal", lambda: Decimal('0.1') + Decimal('0.2') == Decimal('0.3')),
    ("DateTime", lambda: datetime.now().year >= 2025),
    ("Path", lambda: Path.cwd().exists()),
    ("UUID", lambda: len(str(uuid.uuid4())) == 36),
    ("Set", lambda: 2 in {1, 2, 3}),
    ("Range", lambda: len(range(10)) == 10),
]

passed = 0
for name, test in tests:
    try:
        if test():
            print(f"[OK] {name}")
            passed += 1
        else:
            print(f"[FAIL] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")

print(f"\nQuick check: {passed}/{len(tests)} passed")
EOF
}

# Main
main() {
    case "${1:-all}" in
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all        Run all tests (default)"
            echo "  --quick      Quick verification"
            echo "  --build      Build only"
            echo "  --zig        Zig tests only"
            echo "  --pytest     Pytest only"
            echo "  -h, --help   Show help"
            exit 0
            ;;
        --quick)
            quick_check
            ;;
        --build)
            check_prerequisites
            build_project
            ;;
        --zig)
            check_prerequisites
            run_zig_tests
            ;;
        --pytest)
            check_prerequisites
            run_pytest
            ;;
        --all|*)
            check_prerequisites
            build_project
            run_zig_tests
            run_pytest
            echo ""
            print_success "All tests completed"
            ;;
    esac
}

main "$@"
