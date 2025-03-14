#!/bin/bash

CONDAENV=$(conda info | grep "active environment" | awk '{print $NF}')

if [[ "$CONDAENV" != "parl_speech_actors" ]]; then
    echo "Error: Active environment is not 'parl_speech_actors'."
    exit 1
fi

mkdir -p logs

# run in background
nohup ./run_mention_pair_classification_workflow_ollama.sh > "logs/mention_pair_classification_workflow_$(date '+%Y-%m-%dT%H:%M:%S').log" 2>&1 &
