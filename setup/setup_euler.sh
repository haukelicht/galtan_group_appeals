#!/bin/bash
module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy

# cd ..

# create a virtual environment
python3 -m venv --system-site-packages venv 
source venv/bin/activate

# install required packages
pip install numpy==1.25.2 pandas==2.0.3 --upgrade
# ... for sentence segmentation
pip install nltk==3.8.1 stanza==1.6.1 sacremoses==0.1.1 --upgrade
# ... for informativeness ranking
pip install tokenizers==0.15.2 sentencepiece==0.1.99 transformers==4.38.2 sentence-transformers==2.6.1 --upgrade

python3 -c "import torch; print(torch.__version__)"
