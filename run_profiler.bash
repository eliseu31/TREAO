#!/usr/bin/env bash
MACHINE_NAME=$1
IDX_TRIALS=$2
END_TRIALS=$3

for (( i = IDX_TRIALS; i < END_TRIALS; i++ )); do
    python profiler/process_profiler.py "$MACHINE_NAME" "$i"
    sleep 1
done


