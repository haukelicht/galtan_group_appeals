#!/bin/bash

CONDAENV=$(conda info | grep "active environment" | awk '{print $NF}')

if [[ "$CONDAENV" != "galtan_group_appeals" ]]; then
    echo "Error: Active environment is not 'galtan_group_appeals'."
    exit 1
fi

mkdir -p logs

# nohup ./03-mention-attribute-dimension-classification_setfit_model_selection.sh > "logs/mention_attribute_dimension_classification.log" 2>&1 &
# nohup ./03-mention-noneconomic-attributes-classification_setfit_model_selection.sh > "logs/mention_noneconomic_attributes_classification.log" 2>&1 &
# nohup ./03-mention-economic-attributes-classification_setfit_model_selection.sh > "logs/mention_economic_attributes_classification.log" 2>&1 &
