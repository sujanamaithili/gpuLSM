#!/bin/bash

# Exit if any command fails
set -e

# Compiler and flags
NVCC="nvcc"
CXX_FLAGS="-std=c++11 -lcudart"

# Directories (relative to the current script location)
SRC_DIR="../src"
KERNEL_DIR="../kernel"
TEST_DIR="."

# Output binary name
OUTPUT="test_lsm"

# Change to the test directory
cd "$(dirname "$0")"

# Compilation
echo "Compiling LSM tree and test files..."
$NVCC $CXX_FLAGS -I$SRC_DIR -I$KERNEL_DIR -I$INCLUDE_DIR $SRC_DIR/*.cu $KERNEL_DIR/*.cu $TEST_DIR/test.cu -o $OUTPUT

# Run the test
echo "Running tests..."
./$OUTPUT

# Clean up the binary after the test run
echo "Cleaning up..."
rm -f $OUTPUT

echo "Tests completed successfully!"

