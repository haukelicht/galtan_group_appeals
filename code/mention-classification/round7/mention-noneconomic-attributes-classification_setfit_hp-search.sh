#!/bin/bash

BASE_PATH="./../../../"
export PYTHONPATH="${BASE_PATH}:$PYTHONPATH"

STEP="hp_search"
SCRIPT_NAME="./../hp_search_setfit.py"

TASK="noneconomic_attributes_classification"
STRATEGY="mention_text"

# set data dir
DATA_DIR="${BASE_PATH}data/annotations/group_mention_categorization/splits/model_selection"
LABEL_COLS="noneconomic__*"

RESULTS_DIR="${BASE_PATH}results/classifiers/${TASK}/${STEP}/setfit"
#MODELS_SAVE_DIR="${BASE_PATH}models/${TASK}"

# create list of models to try
MODELS=(
    "sentence-transformers/all-mpnet-base-v2"
    "nomic-ai/modernbert-embed-base"
    "google/embeddinggemma-300m"
)

# HP search settings
N_TRIALS=15

declare -A body_batch_sizes
body_batch_sizes=(
    ["sentence-transformers/all-mpnet-base-v2"]="8 16 32"
    ["nomic-ai/modernbert-embed-base"]="8 16 32"
    ["google/embeddinggemma-300m"]="8 16"
)

declare -A head_batch_sizes
head_batch_sizes=(
    ["sentence-transformers/all-mpnet-base-v2"]="4 8 16"
    ["nomic-ai/modernbert-embed-base"]="4 8 16"
    ["google/embeddinggemma-300m"]="4 8"
)

head_learning_rates="0.03 0.01 0.003 0.0001"
l2_weight_decays="0.01 0.015 0.2"
warmup_proportions="0.1 0.15"


OVERWRITE=true

for FOLD in "fold01" "fold02" "fold03" "fold04" "fold05" ; do
    #for FOLD in "fold01" ; do
    FOLD_DIR="${DATA_DIR}/${FOLD}/"

    # iterate over models
    for MODEL_NAME in "${MODELS[@]}"; do
        echo "Finetuning model \"$MODEL_NAME\" with strategy \"$STRATEGY\" on $(basename $FOLD_DIR)"
        
        # create results dir
        RES_DIR="${RESULTS_DIR}/${MODEL_NAME//\//--}/${STRATEGY}/$(basename $FOLD_DIR)/"
        
        if [ "$OVERWRITE" = false ] && [ -d "$RES_DIR" ]; then
            echo "Results directory $RES_DIR already exists and OVERWRITE is set to false. Skipping this run."
            echo " "
            continue
        fi
        
        mkdir -p "$RES_DIR"
        LOG_FILE="${RES_DIR}/hp_search.log"
        CMD_FILE="${RES_DIR}/hp_search.cmd"

        # build the command
        cmd="python3 $SCRIPT_NAME \
            --data_splits_path \"$FOLD_DIR\" \
            --combine_train_val \
            --label_cols \"$LABEL_COLS\" \
            --model_name \"$MODEL_NAME\" \
            --n_trials $N_TRIALS
        "
        
        # use class weighting
        cmd+=" --class_weighting_strategy inverse_proportional "
        
        # get batch size 
        BBS=${body_batch_sizes[$MODEL_NAME]}
        cmd+=" --body_train_batch_sizes $BBS "
        HBS=${head_batch_sizes[$MODEL_NAME]}
        cmd+=" --head_train_batch_sizes $HBS "

        cmd+=" --head_learning_rates $head_learning_rates "
        cmd+=" --l2_weight_decays $l2_weight_decays "
        cmd+=" --warmup_proportions $warmup_proportions "

        cmd+=" --body_early_stopping_patience 5 --body_early_stopping_threshold 0.01"
        cmd+=" --head_early_stopping_patience 5 --head_early_stopping_threshold 0.015"

        # add strategy specific args
        if [ "$STRATEGY" = "span_embedding" ]; then
            cmd+=" --use_span_embeddings "
        elif [ "$STRATEGY" = "concat_prefix" ]; then
            cmd+=" --concat_strategy prefix --concat_sep_token ': ' "
        elif [ "$STRATEGY" = "concat_suffix" ]; then
            cmd+=" --concat_strategy suffix --concat_sep_token ': ' "
        fi

        # add saving args
        cmd+=" --save_results_to \"$RES_DIR\" --overwrite_results"
        # add evaluation args
        cmd+=" --train_best_model   --save_eval_results --save_eval_predictions"

        # TODO: consider final model saving
        # # remove path before / in model name for saving
        # MODEL_NAME_STEM=${MODEL_NAME##*/}
        # MODELPATH="$MODELS_DIR/${MODEL_NAME_STEM}_${STRATEGY}_$(basename $FOLD_DIR)"
        # cmd+=" --train_final_model --save_final_model_to \"$MODELPATH\""

        echo $cmd > "$CMD_FILE"
        echo $cmd
        
        # execute the command
        eval $cmd >> "$LOG_FILE" 2>&1

        echo " "
    done
done

echo "All runs completed."
