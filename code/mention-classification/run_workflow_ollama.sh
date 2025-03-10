#!/bin/bash
# This script runs the zero-shot ICL group mention pair classification  workflow

# conda activate parl_actor_mentions

ts() { date '+%Y-%m-%d %H:%M:%S'; }
message() { echo -e "[$(ts)] $1"; }

# check that ollama is running
if ! pgrep -f "ollama" > /dev/null
then
    # echo "ollama is not running"
    # exit 1
    ollama serve
fi

# hyperparameters
llms=("phi4:14b" "mistral-small:24b" "qwen2.5:32b" "llama3.3:70b")

# globals variables
data_path="./../../data/annotations/group_mention_categorization/social-group-mentions-pair-classification"
input_file="${data_path}/sample.tsv"

for llm in ${llms[@]}; do
        output_file="${data_path}/${llm}_annotations.jsonl"
        if [ -e "$output_file" ]; then
            echo "$output_file exists, skipping..."
            continue
        fi
        message "Running inference with $llm"
        python mention_pair_classification_workflow.py \
            --input_file $input_file \
            --llm $llm \
            --output_file $output_file --overwrite_output_file
            # > /dev/null 2>&1
done
message "Done!"
