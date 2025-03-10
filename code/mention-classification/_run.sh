#!/bin/bash

CONDAENV=$(conda info | grep "active environment" | awk '{print $NF}')

if [[ "$CONDAENV" != "parl_speech_actors" ]]; then
    echo "Error: Active environment is not 'parl_speech_actors'."
    exit 1
fi

mkdir -p logs

# run in background
nohup ./run_workflow_ollama.sh > "logs/workflow_$(date '+%Y-%m-%dT%H:%M:%S').log" 2>&1 &
