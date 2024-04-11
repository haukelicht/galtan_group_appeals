import os

import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
from reconstruction_loss_ranker import ReconstructionLossRanker

from utils import log, clean_memory

from tqdm.auto import tqdm
from typing import List, Union

def embed_and_rank(
    texts: List[str],
    text_ids: List[Union[str, int]],
    embedding_model: SentenceTransformer,
    embedding_batch_size: int=128,
    device: Union[str, torch.device]='cpu',
    epochs: int=25_000,
    seed: int=42,
    return_texts: bool=False,
    sort_by_informativeness: bool=True,
    verbose: bool=True
  ) -> pd.DataFrame:

  n_ = len(texts)

  # embed
  if verbose: log(f'Embedding {n_} texts')
  embeddings = embedding_model.encode(texts, show_progress_bar=verbose, convert_to_tensor=True, normalize_embeddings=True, batch_size=embedding_batch_size)

  # rank
  if verbose: log('Learning ranking from embeddings')
  ranker = ReconstructionLossRanker(hdim=embeddings.shape[0], num_epochs=epochs, log_n_steps=5_000, device=device, seed=seed)
  idxs, _ = ranker.fit(data=embeddings, verbose=verbose)
  ranker.cpu();
  
  del ranker
  clean_memory(device=device)

  # return
  out = {'text_id': text_ids}
  if return_texts:
    out['text'] = texts
  out['informativeness_rank'] = idxs+1
  out = pd.DataFrame(out)
  if sort_by_informativeness:
    out.sort_values('informativeness_rank', inplace=True).reset_index(drop=True, inplace=True)
  return out

def main(args):

  # -- Validate inputs -----

  # check that input_file exists
  assert os.path.exists(args.input_file), ValueError(f'Input file "{args.input_file}" not found.')
  # check that input_file is a file
  assert os.path.isfile(args.input_file), ValueError(f'Input file "{args.input_file}" is not a file.')
  # check that input_file is a csv, tsv, or tab file
  allowed_extensions = ['.csv', '.tsv', '.tab']
  assert any(args.input_file.lower().endswith(ext) for ext in allowed_extensions), ValueError(f'Input file "{args.input_file}" is not a .csv, .tsv, or .tab file.')

  # check output_file
  if args.output_file is not None:
      assert not os.path.exists(args.output_file) or args.overwrite_output_file, ValueError(f'Output file "{args.output_file}" already exists. Use --overwrite_output_file to overwrite.')
  else:
      assert args.overwrite_output_file, ValueError(f'No output file specified. Set --overwrite_output_file to overwrite input file.')

  # check device
  device = args.device if args.device is not None else 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
  if args.verbose: log(f'Using device "{device}"')

  # -- Prepare the data -----

  # read the data
  ext = os.path.splitext(args.input_file)[1]
  sep = ',' if ext=='.csv' else '\t' if ext in ['.tsv', '.tab'] else None
  if args.verbose: log(f'Reading data from file "{args.input_file}"')
  try:
    df = pd.read_csv(args.input_file, sep=sep)
  except Exception as e:
    log('ERROR: could not read data from input file')
    raise e

  # check that columns exist
  cols = [args.text_col, args.id_col]
  if args.group_by is None:
    df['__group__'] = 0
    group_by_cols = ['__group__']
  else:
    group_by_cols = [col.strip() for col in args.group_by.split(',')]
  cols += group_by_cols
  assert all(col in df.columns for col in cols), ValueError(f'Data frame must have columns {cols}')

  # subset to columns
  df = df[cols]

  # remove empty texts
  df = df[~df[args.text_col].isna()]
  df = df[~df[args.text_col].str.match('^\s+$')]

  # -- embedding texts and fitting the ranking model -----
  if args.verbose: log(f'(Down)loading embedding model')
  model = SentenceTransformer(args.embedding_model, device=device)

  # count number of groups
  n_groups = df.groupby(group_by_cols).ngroups
  ranked = []
  for g, d in tqdm(df.groupby(group_by_cols), total=n_groups, desc='Processing groups'):
    try:
      tmp = embed_and_rank(
        texts=d[args.text_col].to_list(),
        text_ids=d[args.id_col].to_list(),
        embedding_model=model,
        device=device,
        seed=args.seed,
        epochs=args.epochs,
        sort_by_informativeness=False,
        verbose=args.verbose if n_groups==1 else False
      )
    except Exception as e:
      grp = ', '.join([f'{k}={v}' for k, v in zip(group_by_cols, g)])
      log(f'ERROR: couldn\'t process data fro group {grp}: {str(e)}')
      continue
    
    tmp = pd.concat([d[group_by_cols], tmp], axis=1)
    ranked.append(tmp)

    if args.max_groups > 0 and len(ranked) >= args.max_groups:
      break

  ranked = pd.concat(ranked, axis=0)
  ranked.rename(columns={'text_id': args.id_col}, inplace=True)

  if '__group__' in ranked.columns:
    ranked.drop(columns='__group__', inplace=True)
  
  # -- write the results -----

  ext = os.path.splitext(args.output_file)[1]
  sep = ',' if ext=='.csv' else '\t' if ext in ['.tsv', '.tab'] else None
  if args.verbose: log(f'Writing results to file "{args.output_file}"')
  try:
    ranked.to_csv(args.output_file, sep=sep, index=False)
  except Exception as e:
    log('ERROR: could not write reasults to output file')
    raise e


if __name__ == '__main__':

  import sys
  import argparse

  desc = [
    'Rank texts according to informativeness based on their text embeddings',
    'appling reconstruction loss ranker proposed in "Selecting More Informative Training Sets with Fewer Observations" (https://doi.org/10.1017/pan.2023.19)'
  ]
  parser = argparse.ArgumentParser(description='\n'.join(desc), formatter_class=argparse.RawTextHelpFormatter)

  # arguments
  parser.add_argument('-i', '--input_file', type=str, help='Input file', required=True)
  parser.add_argument('-o', '--output_file', type=str, help='Output file. If not specified and --overwrite_output_file is set, input file will be overwritten.', required=True)
  parser.add_argument('--overwrite_output_file', action='store_true', help='Overwrite output file if it exists. If output file not specified, input file will be overwritten.')
  parser.add_argument('--text_col', type=str, default='text', help='Name of column containing texts to be translated', required=True)
  parser.add_argument('--id_col', type=str, help='Name of column that containing texts\' IDs', required=True)
  parser.add_argument('--group_by', type=str, help='Name(s) of column(s) to group by')
  parser.add_argument('--max_groups', type=int, default=-1)
  parser.add_argument('--embedding_model', type=str, help='hugging face identifier of embedding model (loaded using SentenceTransformer)', required=True)
  parser.add_argument('--embedding_batch_size', type=int, default=128, help='embedding model encoding batch size')
  parser.add_argument('--device', type=str, default=None, help='device to use for embedding and fitting the ranking model')
  parser.add_argument('--epochs', type=int, default=25_000, help='Number of epochs (forward-backward steps) used to fit the ranking model')
  parser.add_argument('--seed', type=int, default=1234, help='Random seed')
  parser.add_argument('--verbose', action='store_true', help='Print progress bar and other messages.')
  #parser.add_argument('--test', action='store_true', help='Run in test mode.')

  args = parser.parse_args()

  # run main function
  try:
    main(args)
  except Exception as e:
    log(f'ERROR: {str(e)}')
    sys.exit(1)
  else:
    sys.exit(0)
