"""
Group Mention Attributes Classifier

This module provides a flexible classifier for categorizing social group mentions
according to economic and non-economic attributes.

Usage:
    # From Python
    from icl.classifier import GroupMentionClassifier
    
    backend = 'ollama' # alternatively: 'vllm'
    model_id='deepseek-r1:32b' # alternatively: '
    
    classifier = GroupMentionClassifier(
        llm_backend=backend,
        model_name=model_id,
        n_exemplars=5,
        exemplars_file='path/to/exemplars.pkl'
    )
    results = classifier.process_all(df)
    
    # From command line
    python -m icl.classifier \\
        --input data.pkl \\
        --output results.pkl \\
        --llm-backend ollama \\
        --model-name deepseek-r1:32b \\
        --n-exemplars 5 \\
        --exemplars-file train.pkl
"""

import os
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path
from typing import Optional, Literal, Union
import time
import socket
from urllib.parse import urlparse

from icl.definitions import ATTRIBUTES_DEFINITIONS
from icl.workflow import GroupMentionAttributesClassifierWorkflow, ClassificationsResult
from icl._utils.log import log
from llama_index.core.llms.llm import LLM


class GroupMentionClassifier:
    """
    Classifier for social group mention attributes.
    
    Supports multiple LLM backends and both zero-shot and few-shot classification.
    """
    
    def __init__(
        self,
        model_name: str,
        llm_backend: Literal['vllm', 'ollama', 'huggingface', 'openai-like'] = 'vllm',
        temperature: float = 0.0,
        seed: int = 42,
        n_exemplars: Optional[int] = None,
        exemplars_data: Optional[pd.DataFrame] = None,
        exemplars_file: Optional[Union[str, Path]] = None,
        exemplars_sampling_strategy: Literal['random', 'similarity'] = 'similarity',
        exemplars_embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        verbose: bool = False,
        show_progress: bool = True,
        **llm_kwargs
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the model to use
            llm_backend: LLM backend to use ('ollama', 'huggingface', or 'openai-like')
            temperature: Sampling temperature (0.0 for deterministic)
            seed: Random seed for reproducibility
            n_exemplars: Number of exemplars for few-shot learning (None for zero-shot)
            exemplars_data: DataFrame with exemplars (columns: text, mention, attribute labels)
            exemplars_file: Path to file with exemplars (alternative to exemplars_data)
            exemplars_sampling_strategy: 'random' or 'similarity'
            exemplars_embedding_model: Embedding model for similarity retrieval
            verbose: Whether to print verbose output
            show_progress: Whether to show progress bar during prediction
            **llm_kwargs: Additional arguments for the LLM
                For 'openai-like' backend: 
                  - api_base (required): API endpoint URL
                  - api_key (optional): API key, defaults to 'EMPTY'
                  - port_timeout (optional): seconds to wait for port, defaults to 300
        """
        self.llm = self._create_llm(
            backend=llm_backend,
            model_name=model_name,
            temperature=temperature,
            seed=seed,
            **llm_kwargs
        )
        
        self.workflow = GroupMentionAttributesClassifierWorkflow(
            llm=self.llm,
            attributes=ATTRIBUTES_DEFINITIONS,
            n_exemplars=n_exemplars,
            exemplars_data=exemplars_data,
            exemplars_file=exemplars_file,
            exemplars_sampling_strategy=exemplars_sampling_strategy,
            exemplars_embedding_model=exemplars_embedding_model,
            verbose=verbose,
        )
        
        self.verbose = verbose
        self.show_progress = show_progress
    
    @staticmethod
    def _wait_for_port(api_base: str, timeout: int = 300, verbose: bool = False) -> None:
        """
        Wait for API port to be accessible.
        
        Args:
            api_base: API base URL (e.g., 'http://localhost:8000/v1')
            timeout: Maximum time to wait in seconds (default: 5 minutes)
            verbose: Whether to print status messages
            
        Raises:
            TimeoutError: If port is not accessible within timeout
        """
        parsed = urlparse(api_base)
        host = parsed.hostname or 'localhost'
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        
        if verbose:
            log(f"Waiting for API at {host}:{port}...")
        
        start_time = time.time()
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                if verbose:
                    log(f"✓ API port is accessible at {host}:{port}")
                return
            
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"API at {api_base} did not become accessible within {timeout}s. "
                    f"Please ensure the server is running."
                )
            
            time.sleep(1)
    
    @staticmethod
    def _create_llm(
        backend: str,
        model_name: str,
        temperature: float = 0.0,
        seed: int = 42,
        **kwargs
    ) -> LLM:
        """Create LLM instance based on backend."""
        if backend == 'ollama':
            from llama_index.llms.ollama import Ollama
            # Set default request_timeout if not provided to avoid ReadTimeout errors
            if 'request_timeout' not in kwargs:
                kwargs['request_timeout'] = 300.0  # 5 minutes default
            return Ollama(
                model=model_name,
                temperature=temperature,
                seed=seed,
                json_mode=True,
                **kwargs
            )
        elif backend in ('vllm', 'openai-like'):
            from llama_index.llms.openai_like import OpenAILike
            if 'api_base' not in kwargs:
                raise ValueError("api_base is required for OpenAI-like backend")
            api_base = kwargs.pop('api_base')
            api_key = kwargs.pop('api_key', 'EMPTY')
            verbose = kwargs.pop('verbose', False)
            timeout = kwargs.pop('port_timeout', 300)
            
            # Wait for port to be accessible
            GroupMentionClassifier._wait_for_port(api_base, timeout=timeout, verbose=verbose)
            
            llm = OpenAILike(
                model=model_name,
                api_base=api_base,
                api_key=api_key,
                temperature=temperature,
                **kwargs
            )
            if backend == 'vllm':
                llm.tokenizer = model_name
            return llm
            
        elif backend == 'huggingface':
            from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
            if "HUGGINGFACE_API_TOKEN" not in os.environ:
                raise ValueError("HUGGINGFACE_API_TOKEN environment variable must be set for HuggingFace backend")
            return HuggingFaceInferenceAPI(
                model_name=model_name,
                token=os.getenv("HUGGINGFACE_API_TOKEN"),
                temperature=temperature,
                num_output=kwargs.get('max_tokens', 1024),
                **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
            )

            # TODO: consider checking API can be reacehed here too
        else:
            raise ValueError(f"Unsupported LLM backend: {backend}. Choose 'ollama', 'huggingface', or 'openai-like'")
        
        return ll
    
    async def process_one(
        self,
        text: str,
        mention: str,
        **mention_kwargs
    ) -> ClassificationsResult:
        """
        Classify a single mention.
        
        Args:
            text: The sentence/text containing the mention
            mention: The specific mention to classify
            **mention_kwargs: Additional metadata to store with the result
            
        Returns:
            ClassificationsResult with annotations for all attributes
        """
        return await self.workflow.run(text=text, mention=mention)
    
    async def process_all(
        self,
        data: pd.DataFrame,
        id_col: str = 'mention_id',
        text_col: str = 'text',
        mention_col: str = 'mention',
    ) -> pd.DataFrame:
        """
        Classify mentions in a DataFrame.
        
        Args:
            data: DataFrame with mentions to classify
            id_col: Column name for mention IDs
            text_col: Column name for text/sentences
            mention_col: Column name for mentions
            
        Returns:
            DataFrame with classification results
        """
        # Deduplicate to avoid processing same mention multiple times
        data_dedup = data[[id_col, text_col, mention_col]].drop_duplicates().reset_index(drop=True)
        
        annotations = {}
        error_count = 0
        async for _, row in tqdm_asyncio(
            data_dedup.iterrows(), 
            total=len(data_dedup),
            desc="Classifying mentions",
            disable=not self.show_progress
        ):
            try:
                annotations[row[id_col]] = await self.workflow.run(
                    text=row[text_col], 
                    mention=row[mention_col]
                )
            except Exception as e:
                error_count += 1
                # Always log errors, not just in verbose mode
                log(f"ERROR processing {id_col} {row[id_col]}: {e}")
                # Store empty result on error
                annotations[row[id_col]] = ClassificationsResult(
                    text=row[text_col], 
                    mention=row[mention_col]
                )
        
        # Convert to DataFrame
        results = pd.concat({
            mid: res.to_pandas() 
            for mid, res in annotations.items()
        }).reset_index(level=0, names=[id_col])
        
        # Log error summary if any errors occurred
        if error_count > 0:
            log(f"⚠️  WARNING: {error_count}/{len(data_dedup)} mentions failed during classification")
            log(f"   Failed mentions will have missing attribute_id, classification, and reasoning columns")
        
        return results
    
    def process_sync(self, *args, **kwargs) -> pd.DataFrame:
        """Synchronous wrapper for predict() method. Works in both scripts and Jupyter notebooks."""
        import asyncio
        
        # Check if we're in a running event loop (e.g., Jupyter notebook)
        try:
            loop = asyncio.get_running_loop()
            # In Jupyter/IPython - use nest_asyncio to allow nested event loops
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                raise RuntimeError(
                    "Running in a Jupyter notebook requires 'nest_asyncio'. "
                    "Install it with: pip install nest_asyncio"
                )
            return asyncio.run(self.process_all(*args, **kwargs))
        except RuntimeError:
            # No running loop, regular script/CLI
            return asyncio.run(self.process_all(*args, **kwargs))


def main():
    """Command-line interface for the classifier."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Classify economica and non-economic attributes expressed in social group mentions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', required=True, help='Input file (pkl, csv, tsv, jsonl)')
    parser.add_argument('--output', '-o', required=True, help='Output file (pkl, csv, tsv, jsonl)')
    
    # LLM configuration
    parser.add_argument('--llm-backend', '-b', required=True, choices=['vllm', 'ollama', 'huggingface', 'openai-like'],
                        help='LLM backend to use')
    parser.add_argument('--api-base', default='http://localhost:8000/v1',
                        help='API base URL for openai-like backend (d, http://localhost:8000/v1)')
    parser.add_argument('--api-key', default='EMPTY', help='API key for openai-like backend')
    parser.add_argument('--port-timeout', type=int, default=300, 
                        help='Timeout in seconds for waiting for API port to be accessible')
    
    parser.add_argument('--model-name', '-m', required=True, help='Model name')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    
    # Few-shot configuration
    parser.add_argument('--n-exemplars', '-k', type=int, default=None,
                        help='Number of exemplars for few-shot learning (None for zero-shot)')
    parser.add_argument('--exemplars-file', help='File with exemplars for few-shot learning')
    parser.add_argument('--exemplars-sampling-strategy', default='similarity',
                        choices=['random', 'similarity'], help='Exemplar sampling strategy')
    parser.add_argument('--exemplars-embedding-model', default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Embedding model for similarity-based retrieval')
    
    # Data columns
    parser.add_argument('--id-col', default='mention_id', help='Column name for mention IDs')
    parser.add_argument('--text-col', default='text', help='Column name for text')
    parser.add_argument('--mention-col', default='mention', help='Column name for mentions')
    
    # Other
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--show-progress', action='store_true', default=True, help='Show progress bar')
    parser.add_argument('--no-progress', dest='show_progress', action='store_false', help='Hide progress bar')
    parser.add_argument('--test', type=int, default=None, metavar='N',
                        help='Test mode: sample N examples from input data (default: process all)')
    
    args = parser.parse_args()
    
    # Load input data
    input_path = Path(args.input)
    ext = input_path.suffix.lstrip('.')
    if ext == 'pkl':
        data = pd.read_pickle(input_path)
    elif ext in ('csv', 'tsv', 'tab'):
        sep = '\t' if ext in ('tsv', 'tab') else ','
        data = pd.read_csv(input_path, sep=sep)
    elif ext == 'jsonl':
        data = pd.read_json(input_path, lines=True)
    else:
        raise ValueError(f"Unsupported input format: {ext}")
    
    if args.verbose:
        log(f"Loaded {len(data)} rows from {input_path}")
    
    # Sample data if in test mode
    if args.test is not None:
        if args.test > len(data):
            if args.verbose:
                log(f"Warning: --test={args.test} exceeds data size ({len(data)}), using all data")
        else:
            data = data.sample(n=args.test, random_state=args.seed).reset_index(drop=True)
            if args.verbose:
                log(f"Test mode: sampled {len(data)} examples")
    
    # Validate openai-like backend requirements
    if args.llm_backend == 'openai-like' and not args.api_base:
        parser.error('--api-base is required when using openai-like backend')
    
    # Initialize classifier
    llm_kwargs = {}
    if args.llm_backend == 'openai-like':
        llm_kwargs['api_base'] = args.api_base
        llm_kwargs['api_key'] = args.api_key
        llm_kwargs['port_timeout'] = args.port_timeout
        llm_kwargs['verbose'] = args.verbose
    
    classifier = GroupMentionClassifier(
        llm_backend=args.llm_backend,
        model_name=args.model_name,
        api_base=args.api_base,
        temperature=args.temperature,
        seed=args.seed,
        n_exemplars=args.n_exemplars,
        exemplars_file=args.exemplars_file,
        exemplars_sampling_strategy=args.exemplars_sampling_strategy,
        exemplars_embedding_model=args.exemplars_embedding_model,
        verbose=args.verbose,
        show_progress=args.show_progress,
        **llm_kwargs
    )
    
    mode = "few-shot" if args.n_exemplars else "zero-shot"
    if args.verbose:
        log(f"Running {mode} classification with {args.llm_backend}: '\033[1m{args.model_name}\033[0m'")
    
    # Run predictions
    results = classifier.process_sync(
        data=data,
        id_col=args.id_col,
        text_col=args.text_col,
        mention_col=args.mention_col,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lstrip('.')
    
    if ext == 'pkl':
        results.to_pickle(output_path)
    elif ext in ('csv', 'tsv', 'tab'):
        sep = '\t' if ext in ('tsv', 'tab') else ','
        results.to_csv(output_path, sep=sep, index=False)
    elif ext == 'jsonl':
        results.to_json(output_path, lines=True, orient='records')
    else:
        raise ValueError(f"Unsupported output format: {ext}")
    
    if args.verbose:
        log(f"Saved {len(results)} results to {output_path}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        import sys
        sys.exit(130)
