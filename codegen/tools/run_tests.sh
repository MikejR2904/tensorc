#!/bin/bash
# codegen/tools/run_tests.sh
# Quick script to build and run all codegen tests

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║ TensorC Codegen Test Suite - Build & Run                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# Build
echo ""
echo "📦 Building tests..."
cd "$PROJECT_ROOT/build"
cmake --build . --target codegen-scalar-test codegen-legacy-extended-test codegen-progressive-test --config Debug

# Run tests
echo ""
echo "🧪 Running Scalar Operations Tests..."
./bin/codegen-scalar-test.exe

echo ""
echo "🧪 Running Extended Operations Tests..."
./bin/codegen-legacy-extended-test.exe

echo ""
echo "🧪 Running Progressive Lowering Tests..."
./bin/codegen-progressive-test.exe

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║ All Tests Completed ✓                                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
