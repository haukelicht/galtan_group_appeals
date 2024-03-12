import warnings
warnings.filterwarnings("ignore")
    
import os

import torch
from sacremoses import MosesPunctNormalizer
import stanza

from tqdm.auto import tqdm
from typing import List

from datetime import datetime

ts = lambda : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log = lambda x: print(f'[{ts()}]', x)

def main(args):
    
    # read text in lines in input file (removing trailing line breaks)
    with open(args.input_file, 'r') as file:
        lines = [line.removesuffix('\n') for line in file.readlines()]
    if args.test:
        lines = lines[:args.n_test]
    n = len(lines)
    log(f'INFO: processing {n} lines in file "{args.input_file}"')

    # init normalizer and tokenizer
    normalizer = MosesPunctNormalizer(
        lang=args.language, 
        pre_replace_unicode_punct=True, 
        post_remove_control_chars=True,
        norm_quote_commas=True,
        norm_numbers=True,
    )
    tokenizer = stanza.Pipeline(lang=args.language, processors='tokenize', verbose=False, device=args.device)
    tokenizer.processors['tokenize'].config['max_seqlen'] = 10_000

    lines = normalizer.normalize(lines)

    sentences = []
    for i in tqdm(range(0, n, args.batch_size), total=n//args.batch_size, desc='Splitting texts into sentences'):
        batch = lines[i:i+args.batch_size]
        batch = tokenizer.bulk_process(batch)
        batch = [sent.text for doc in batch for sent in doc.sentences]
        sentences += batch
    
    log(f'INFO: writing results to "{args.output_file}"')
    with open(args.output_file, 'w') as file:
        file.write('\n'.join(sentences))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tokenize text with stanza')
    parser.add_argument('--input_file', type=str, help='Path of input file')
    parser.add_argument('--language', type=str, help='Language code')
    parser.add_argument('--output_file', type=str, help='Path of output file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output file')
    parser.add_argument('--device', type=str, default=None, help='Device to use', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--test', action='store_true', help='Run test')
    parser.add_argument('--n_test', type=int, default=10, help='Run test')

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        log('ERROR: Input file not found')
    if os.path.isfile(args.output_file) and not args.overwrite:
        log('ERROR: Output file already exists')
    
    try:
        _ = stanza.Pipeline(lang=args.language, processors='tokenize', verbose=False, device='cpu')
    except:
        log(f'ERROR: No stanza tokenizer available for language "{args.language}"')
    
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    main(args)

