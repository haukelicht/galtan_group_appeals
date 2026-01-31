from pydantic import BaseModel, Field
from typing import Literal, Optional

from llama_index.core.program import LLMTextCompletionProgram 
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.output_parsers import PydanticOutputParser

from icl.prompts import ATTRIBUTE_CLASSIFICATION_TASK_PROMPT_TEMPLATE
from copy import deepcopy

# for few-shot
import pandas as pd
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from icl.retrieval import RandomRetriever, retrieve_mention_exemplars


class HasAttribute(BaseModel):
    reasoning: str = Field(..., description="Sentence explaining the reasoning behind the decision")
    classification: Literal["yes", "no"] = Field(..., description="Decision whether the mention extracted from the text captures a _social_ group mention")


class AttributeClassificationProgram(LLMTextCompletionProgram):
    prompt_template = PromptTemplate(ATTRIBUTE_CLASSIFICATION_TASK_PROMPT_TEMPLATE)
    output_cls = HasAttribute
    template_vars = ['text', 'mention']
    
    def __init__(
            self, 
            llm: LLM, 
            verbose: bool=False,
            n_exemplars: Optional[int]=None,
            exemplars_file: str=None,
            exemplars_text_col: str='text',
            exemplars_mention_col: str='mention',
            exemplars_label_col: str='label',
            exemplars_sampling_strategy: Literal['random', 'similarity']='similarity',
            exemplars_embedding_engine: Literal['ollama', 'hf']='hf',
            exemplars_embedding_model: str=None,
        ):
        """
        Args:
            llm (LLM): The LLM to use for text completion.
            verbose (bool): Whether to print verbose output.
            n_exemplars (int, optional): 
                Number of exemplars to use for few-shot classification. If None, zero-shot classification is used (i.e., no exemplars).
            exemplars_file (str, optional):
                Path to the file containing the exemplars. Must be provided if n_exemplars > 0.
            exemplars_sampling_strategy (str): 
                Strategy for retrieving exemplars. Must be one of ['random', 'similarity'].
            exemplars_embedding_engine (str): 
                Embedding engine to use for similarity-based exemplars retrieval. Must be one of ['ollama', 'hf'].
            exemplars_embedding_model (str): 
                Model name for the embedding engine. Must be provided if ``exemplars_sampling_strategy`` is 'similarity'.
        
        Raises:
            ValueError: If n_exemplars is not None and less than or equal to 0.
            ValueError: If exemplars_file is None and n_exemplars > 0.
            ValueError: If exemplars_sampling_strategy is None and n_exemplars > 0.
            ValueError: If exemplars_sampling_strategy is not one of ['random', 'similarity'].
            ValueError: If exemplars_embedding_engine is None and exemplars_sampling_strategy is 'similarity'.
            ValueError: If exemplars_embedding_engine is not one of ['ollama', 'hf'].
            ValueError: If the exemplars file format is not supported.
            ValueError: If the exemplars file is empty.
            ValueError: If the exemplars file has less than n_exemplars exemplars.
        
        """

        # instantiate the LLMTextCompletionProgram
        super().__init__(
            llm=llm, 
            prompt=self.prompt_template, 
            output_cls=self.output_cls, 
            output_parser=PydanticOutputParser(output_cls=self.output_cls),
            verbose=verbose
        )

        # setup few-shot logic
        self.is_fewshot = False
        self.k_shots = False
        if n_exemplars and n_exemplars <= 0:
            raise ValueError("n_exemplars must be > 0 for few-shot classification or `None` (zero-shot)")
        elif n_exemplars and n_exemplars > 0:
            self.is_fewshot = True 
            self.k_shots = n_exemplars
            
            if exemplars_file is None:
                raise ValueError("exemplars_file must be provided if n_exemplars > 0")
            
            if exemplars_sampling_strategy is None:
                raise ValueError("exemplars_sampling_strategy must be provided if n_exemplars > 0")
            if exemplars_sampling_strategy not in ['random', 'similarity']:
                raise ValueError(f"exemplars_sampling_strategy must be one of ['random', 'similarity'], but got {exemplars_sampling_strategy}")
            else: 
                self.exemplars_sampling_strategy = exemplars_sampling_strategy
            
            # try loading the exemplars file
            ext = exemplars_file.split('.')[-1]
            usecols = [exemplars_text_col, exemplars_mention_col, exemplars_label_col]
            if ext in ('tsv', 'tab'):
                exemplars = pd.read_csv(exemplars_file, sep='\t', usecols=usecols)
            elif ext in ('csv'):
                exemplars = pd.read_csv(exemplars_file, usecols=usecols)
            elif ext == 'jsonl':
                exemplars = pd.read_json(exemplars_file, lines=True)
                # exemplars = [{k: ex[k] for k in usecols} for ex in exemplars]
            else:
                raise ValueError(f"Unsupported exemplars file format: {ext}")
            
            assert len(exemplars) > 0, f"exemplars file {exemplars_file} is empty"
            assert len(exemplars) >= n_exemplars, f"exemplars file {exemplars_file} has less than {n_exemplars} exemplars"
            
            # convert to text nodes for index
            exemplars = [
                TextNode(
                    text=ex[exemplars_mention_col], 
                    metadata={
                        'classification': 'yes' if ex[exemplars_label_col]==1 else 'no', 
                        'text': ex[exemplars_text_col]
                    }
                )
                for _, ex in exemplars.iterrows()
            ]

            # initialize the retriever
            if self.exemplars_sampling_strategy == 'random':
                # random (local) retrievers samples n_exemplars at random for each input
                self.retriever = RandomRetriever(exemplars, k=self.k_shots)
            elif self.exemplars_sampling_strategy == 'similarity':
                if exemplars_embedding_engine is None:
                    raise ValueError("exemplars_embedding_engine must be provided if n_exemplars > 0")
                elif exemplars_embedding_engine == 'ollama':
                    encoder = OllamaEmbedding(exemplars_embedding_model)
                elif exemplars_embedding_engine == 'hf':
                    encoder = HuggingFaceEmbedding(exemplars_embedding_model)
                else:
                    raise ValueError(f"Unsupported embedding engine: {exemplars_embedding_engine}")
                if verbose:
                    print(f'Embedding {len(exemplars)} exemplars using "{exemplars_embedding_model}" with {exemplars_embedding_engine}')
                self.retriever = VectorStoreIndex(exemplars, embed_model=encoder).as_retriever(similarity_top_k=self.k_shots*3)
                if exemplars_embedding_engine and exemplars_embedding_engine == 'hf':
                    encoder._model.to('cpu'); # free up GPU memory
                del encoder
        
            # add an exemplars placeholder to the prompt template
            prompt_text = deepcopy(ATTRIBUTE_CLASSIFICATION_TASK_PROMPT_TEMPLATE)
            prompt_text += '\n\n## Examples\n\n{exemplars}'
            prompt_text += '\n\n## Input\n\nText: """{text}"""\nMention: """{mention}"""'
            # update the prompt template
            self.prompt = PromptTemplate(prompt_text, function_mappings={"exemplars": self._retrieve_mention_classification_exemplars})

    def _retrieve_mention_classification_exemplars(self, mention, verbose: bool=False, **kwargs):
        if not self.is_fewshot:
            raise ValueError("Few-shot exemplars not set up")
        res = retrieve_mention_exemplars(
            retriever=self.retriever, 
            k=self.k_shots,
            strategy=self.exemplars_sampling_strategy,
            mention=mention, # <== input text used for retrieval passed to kwargs of ``retrieve_mention_exemplars()``
            text_field_name='text',
            mention_field_name='mention',
            format_metadata_fun=lambda text, metadata: self.output_cls(reasoning='...', classification=metadata['classification']),
            return_as='text',
            # input_key_name='INPUT',
            # response_key_name='RESPONSE'
        )
        if verbose:
            print( '--- retrieved exemplars ---')
            print(res)
            print( '---------------------------')
        return res
