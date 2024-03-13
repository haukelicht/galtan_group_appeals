#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hauke.licht@wiso.uni-koeln.de
#SBATCH --job-name=split_manifestos_into_sentences
#SBATCH --output=%x.log
#SBATCH --error=%x.err

module load gcc/8.2.0 eth_proxy python_gpu/3.11.2
# source ./../../venv/bin/activate

# ParlSpeech2 based corpora 
INPUTPATH='../../data/manifestos/raw'
OUTPUTPATH='../../data/manifestos/sentences'

# list .txt files in INPUTPATH recursively and iterate over them
for input_file in $(find ${INPUTPATH} -name '*.txt' | head -n 3); do
    
    output_file="$OUTPUTPATH/${file#${INPUTPATH}}"
    # get the dirnmae of output_file and create it recursively
    mkdir -p $(dirname $output_file)

    # get the part behind - and before .txt in the file name to use as language code
    lang=$(basename $input_file | sed -n 's/.*-\([a-z-]*\)\.txt/\1/p')
    
    echo $lang
    # # split sentences
    # python3 sentence_split.py \
    #     --input_file $input_file \
    #     --language $lang \
    #     --output_file $output_file \
    #     --overwrite \
    #     --batch_size 512
done 
