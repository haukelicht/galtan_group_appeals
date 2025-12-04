#!/bin/bash

export PYTHONPATH="./../../../:$PYTHONPATH"

# get name of current file
SCRIPT_NAME="./../finetune_setfit.py"

# set data dir
TASK="attribute_dimension_classification"
LABEL_COLS="economic,noneconomic"
STEP="model_selection"

DATA_DIR="../../../data/annotations/group_mention_categorization/${STEP}/splits"
RESULTS_DIR="../../../results/classifiers/${TASK}/${STEP}/setfit"

# create list of models to try
MODELS=(
    "sentence-transformers/all-mpnet-base-v2"
    "sentence-transformers/all-MiniLM-L6-v2"
    "nomic-ai/modernbert-embed-base"
    "Alibaba-NLP/gte-modernbert-base"
    "google/embeddinggemma-300m"
    "Qwen/Qwen3-Embedding-0.6B"
)

declare -A batch_sizes
batch_sizes=(
    ["sentence-transformers/all-mpnet-base-v2"]="32 16"
    ["sentence-transformers/all-MiniLM-L6-v2"]="32 16"
    ["nomic-ai/modernbert-embed-base"]="32 16"
    ["Alibaba-NLP/gte-modernbert-base"]="32 16"
    ["ibm-granite/granite-embedding-english-r2"]="32 16"
    ["google/embeddinggemma-300m"]="16 8"
    ["Qwen/Qwen3-Embedding-0.6B"]="8 4"
)


# create strategy list
STRATEGIES=(
    "mention_text"
    "span_embedding"
    "concat_prefix"
    "concat_suffix"
)

declare -A skip_strategy
skip_strategy=(
    # can't use span_embedding strategy because it uses `pooling_mode_cls_token`
    ["Alibaba-NLP/gte-modernbert-base"]="span_embedding"
    ["ibm-granite/granite-embedding-english-r2"]="span_embedding"
)


# get folds folders
FOLDS_FOLDERS=$(ls -d ${DATA_DIR}/fold*/)

runs=0
max_runs=null # NOTE: we have 6 (models) x 4 (strategies) x 5 (folds) = 120 runs in total, set to null to run all

OVERWRITE=true

# iterate over models
for MODEL_NAME in "${MODELS[@]}"; do
    for STRATEGY in "${STRATEGIES[@]}"; do
        
        # check if we need to skip this strategy for this model
        if [ "${skip_strategy[$MODEL_NAME]}" = "$STRATEGY" ]; then
            echo "Skipping strategy \"$STRATEGY\" for model \"$MODEL_NAME\" as per skip list."
            echo " "
            continue
        fi

        for FOLD_DIR in ${FOLDS_FOLDERS}; do
            echo "Finetuning model \"$MODEL_NAME\" with strategy \"$STRATEGY\" on $(basename $FOLD_DIR)"
            
            # create results dir
            RES_DIR="${RESULTS_DIR}/${MODEL_NAME//\//--}/${STRATEGY}/$(basename $FOLD_DIR)/"
            
            if [ "$OVERWRITE" = false ] && [ -d "$RES_DIR" ]; then
                echo "Results directory $RES_DIR already exists and OVERWRITE is set to false. Skipping this run."
                echo " "
                continue
            fi
            
            mkdir -p "$RES_DIR"
            LOG_FILE="${RES_DIR}/finetuning.log"
            CMD_FILE="${RES_DIR}/finetuning.cmd"

            # build the command
            cmd="python3 $SCRIPT_NAME \
                --data_splits_path \"$FOLD_DIR\" \
                --label_cols \"$LABEL_COLS\" \
                --model_name \"$MODEL_NAME\"
            "
            
            # get batch size 
            BS=${batch_sizes[$MODEL_NAME]}
            cmd+=" --train_batch_sizes $BS "
            
            # use class weighting
            cmd+=" --class_weighting_strategy inverse_proportional "

            # add strategy specific args
            if [ "$STRATEGY" = "span_embedding" ]; then
                cmd+=" --use_span_embeddings "
            elif [ "$STRATEGY" = "concat_prefix" ]; then
                cmd+=" --concat_strategy prefix --concat_sep_token ': ' "
            elif [ "$STRATEGY" = "concat_suffix" ]; then
                cmd+=" --concat_strategy suffix --concat_sep_token ': ' "
            fi

            # add evaluation args
            cmd+=" --save_eval_results_to \"$RES_DIR\" --overwrite_results  --do_eval --save_eval_results --save_eval_predictions "

            echo $cmd > "$CMD_FILE"
            
            # execute the command
            eval $cmd >> "$LOG_FILE" 2>&1

            runs=$((runs + 1))
            # test if max runs is set and reached
            if [ $max_runs != null ] && [ $runs -ge $max_runs ]; then
                echo "Reached maximum number of runs ($max_runs). Exiting."
                exit 0
            fi
            echo " "
        done
    done    
done

echo "All runs completed."
