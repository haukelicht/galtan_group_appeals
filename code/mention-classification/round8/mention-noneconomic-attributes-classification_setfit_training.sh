#!/bin/bash

BASE_PATH="./../../.."
export PYTHONPATH="${BASE_PATH}/:$PYTHONPATH"

SCRIPT_NAME="./../finetune_setfit.py"

TASK="noneconomic_attributes_classification"
LABEL_COLS="noneconomic__*"
STRATEGY="mention_text"

# set data dir
DATA_DIR="${BASE_PATH}/data/annotations/group_mention_categorization/splits/model_selection/fold01"
# NOTE: we are going to combine all splits so fold doesn't matter here

BASE_MODEL="sentence-transformers/all-mpnet-base-v2"
MODEL_NAME="all-mpnet-base-v2_noneconomic-attributes-classifier"
MODEL_SAVE_PATH="${BASE_PATH}/models/${MODEL_NAME}"

OVERWRITE=true

if [ "$OVERWRITE" = false ] && [ -d "$MODEL_SAVE_PATH" ]; then
    echo "Model directory $MODEL_SAVE_PATH already exists. Exiting."
    exit 1
fi

echo "Finetuning model \"$BASE_MODEL\" with strategy \"$STRATEGY\" "
        
# build the command
cmd="python3 $SCRIPT_NAME \
    --data_splits_path \"$DATA_DIR\" \
    --combine_splits train val test \
    --label_cols \"$LABEL_COLS\" \
    --model_name \"$BASE_MODEL\" \
    --class_weighting_strategy inverse_proportional \
    --num_epochs 1 4 \
    --train_batch_sizes 32 4 \
    --body_train_max_steps 75 \
    --head_learning_rate 0.010 \
    --l2_weight 0.01 \
    --warmup_proportion 0.15 \
    --save_model \
    --save_model_to \"$MODEL_SAVE_PATH\"
"
echo $cmd

# execute the command
eval $cmd > "$MODEL_NAME.log" 2>&1

echo "Done!"
