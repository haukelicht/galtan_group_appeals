#!/bin/bash
module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy

pip install --user nltk==3.8.1 stanza==1.6.1 sacremoses==0.1.1

python3 -c "import torch; print(torch.__version__)"
