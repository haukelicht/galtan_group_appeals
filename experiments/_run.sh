#!/bin/bash

CONDAENV=$(conda info | grep "active environment" | awk '{print $NF}')

if [[ "$CONDAENV" != "icl_group_attribute_classification" ]]; then
    echo "Error: Active environment is not 'icl_group_attribute_classification'."
    exit 1
fi

mkdir -p logs

# Run the script in background and capture its PID
nohup ./run_10shot.sh > "logs/run_10shot.log" 2>&1 &
SCRIPT_PID=$!

# Save the PID to a file for later reference
echo $SCRIPT_PID > "logs/run_10shot.pid"

echo "Script started with PID: $SCRIPT_PID"
echo "PID saved to: logs/run_10shot.pid"


