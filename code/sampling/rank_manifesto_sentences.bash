pyhton3 informativeness_ranking.py \
    --input_file 'all_manifesto_sentences.tsv' \
    --text_col 'text' \
    --id_col 'sentence_id' \
    --output_file 'all_manifesto_sentences_informativeness_ranking.tsv' --overwrite_output_file \
    --embedding_model 'intfloat/multilingual-e5-large-instruct' \
    --device 'cuda:0' \
    --seed 1234 \
    --epochs 25000 \
    --verbose

