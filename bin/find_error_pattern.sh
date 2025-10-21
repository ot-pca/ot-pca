#!/bin/bash
set -m 

# Define the shared output file and error log
OUTPUT_FILE=./output/error_patterns_hqc128.csv
ERROR_LOG=./logs/error_log.txt

mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$ERROR_LOG")"

# Clear the error log file
> "$ERROR_LOG"

# Store the PIDs of the background processes in an array
pids=()

# Cleanup function to kill all child processes
cleanup() {
    trap - INT TERM EXIT

    echo -e "Interrupt received, killing process groups one by one..."
    if [ ${#pids[@]} -ne 0 ]; then
        for pid in "${pids[@]}"; do
            kill -- -"${pid}"
        done
    fi
    echo "Cleanup finished."
    exit 1
}

# Run the 'cleanup' function if it receives an INT (Ctrl+C), TERM (kill command), or EXIT signal.
trap cleanup INT TERM EXIT

# Function to run iterations independently for each core
run_task() {
    core_id=$1  # Get core id (just for tracking)
    
    # Loop x times independently
    for iteration in {1..5}; do
        echo "Core $core_id: Starting iteration $iteration..."

        # Redirect only stderr to the error log
        python ./src/find_error_pattern.py hqc128 "$OUTPUT_FILE" 2>> "$ERROR_LOG"

        status=$?
        if [ $status -ne 0 ]; then
            echo "Core $core_id: Error occurred in iteration $iteration. Check the error log: $ERROR_LOG"
            exit 1
        fi

        echo "Core $core_id: Iteration $iteration completed successfully."
    done
}

# Run tasks for 3 cores independently in parallel
run_task 1 &
pids+=($!) # Add the PID of the last background process to the array

run_task 2 &
pids+=($!)

run_task 3 &
pids+=($!)

# Wait for all background jobs to finish normally
for pid in "${pids[@]}"; do
    wait "$pid"
done

# If the script finishes without being interrupted, remove the trap so the cleanup function doesn't run on a normal exit.
trap - INT TERM EXIT

echo "All tasks completed successfully."
