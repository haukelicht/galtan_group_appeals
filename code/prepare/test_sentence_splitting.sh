#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hauke.licht@wiso.uni-koeln.de
#SBATCH --job-name=test_sentence_splitting
#SBATCH --output=%x.log
#SBATCH --error=%x.err

module load gcc/8.2.0 eth_proxy python_gpu/3.11.2
# source ./../../venv/bin/activate

# ParlSpeech2 based corpora 
INPUTPATH='../../data/manifestos/raw'
OUTPUTPATH='../../data/manifestos/tests'

mkdir -p $OUTPUTPATH
python3 sentence_split.py \
    --input_file "$INPUTPATH/aus/200410/63110_200410-english.txt" \
    --language 'english' \
    --output_file "$OUTPUTPATH/63110_200410-english.txt" \
    --overwrite \
    --batch_size 512
