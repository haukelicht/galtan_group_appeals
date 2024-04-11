#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40G
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hauke.licht@wiso.uni-koeln.de
#SBATCH --job-name=manifest_sentences_informativeness_ranking_test
#SBATCH --output=%x.log
#SBATCH --error=%x.err

module load gcc/8.2.0 eth_proxy python_gpu/3.11.2
source ./../../venv/bin/activate

DATAPATH='../../data/manifestos'

pyhton3 informativeness_ranking.py \
    --input_file "$DATAPATH/'test_sentences.tsv" \
    --text_col 'text' \
    --id_col 'sentence_id' \
    --output_file "$DATAPATH/test_sentences_informativeness_ranking.tsv" --overwrite_output_file \
    --embedding_model 'intfloat/multilingual-e5-large-instruct' \
    --device 'cuda:0' \
    --seed 1234 \
    --epochs 25000 \
    --verbose

