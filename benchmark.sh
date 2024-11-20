#!/bin/bash

make
cd cpu/

make
cd ../

# GPU and CPU program executables
GPU_EXECUTABLE="./GPULSM"
CPU_EXECUTABLE="./cpuTest"

# Test buffer sizes
BUFFER_SIZES=(16 256 4096 65536 1048576 16777216)

# Output file
OUTPUT_FILE="lsm_test_summary.txt"

# Header for the output table
{
    printf "+-------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+\n"
    printf "| Buffer Size | GPU Init Time (s) | GPU Insert Time (s) | GPU Lookup Time (s) | CPU Init Time (s) | CPU Insert Time (s) | CPU Lookup Time (s) |\n"
    printf "+-------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+\n"
} > $OUTPUT_FILE

# Loop through each buffer size
for BUFFER_SIZE in "${BUFFER_SIZES[@]}"; do
    echo "Running GPU test for buffer size: $BUFFER_SIZE"
    # Run GPU program and capture output
    GPU_OUTPUT=$($GPU_EXECUTABLE -p 4 $BUFFER_SIZE)
    
    # Parse GPU results
    GPU_INIT_TIME=$(echo "$GPU_OUTPUT" | grep "Init time" | awk '{print $3}')
    GPU_INSERT_TIME=$(echo "$GPU_OUTPUT" | grep "Insert time" | awk '{print $3}')
    GPU_LOOKUP_TIME=$(echo "$GPU_OUTPUT" | grep "Lookup time" | awk '{print $3}')

    echo "Running CPU test for buffer size: $BUFFER_SIZE"
    # Run CPU program and capture output
    CPU_OUTPUT=$(cd cpu && $CPU_EXECUTABLE -p 4 $BUFFER_SIZE)
    
    # Parse CPU results
    CPU_INIT_TIME=$(echo "$CPU_OUTPUT" | grep "Init time" | awk '{print $3}')
    CPU_INSERT_TIME=$(echo "$CPU_OUTPUT" | grep "Insert time" | awk '{print $3}')
    CPU_LOOKUP_TIME=$(echo "$CPU_OUTPUT" | grep "Lookup time" | awk '{print $3}')
    
    # Append results to the summary file in formatted columns
    printf "| %11s | %17s | %17s | %17s | %17s | %17s | %17s |\n" \
        "$BUFFER_SIZE" "$GPU_INIT_TIME" "$GPU_INSERT_TIME" "$GPU_LOOKUP_TIME" \
        "$CPU_INIT_TIME" "$CPU_INSERT_TIME" "$CPU_LOOKUP_TIME" >> $OUTPUT_FILE

done

# Add table footer
{
    printf "+-------------+-------------------+-------------------+-------------------+-------------------+-------------------+-------------------+\n"
} >> $OUTPUT_FILE

# Display the summary table
cat $OUTPUT_FILE

