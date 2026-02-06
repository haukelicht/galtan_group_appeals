from icl.prompts import ATTRIBUTE_CLASSIFICATION_TASK_PROMPT_TEMPLATE

from llama_index.core.program import LLMTextCompletionProgram 
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai_like import OpenAILike

from pydantic import BaseModel, Field
from icl._utils.hf_utils import get_response_format
from llama_index.core.output_parsers import PydanticOutputParser
from openai.resources.chat.completions.completions import _type_to_response_format

import pandas as pd
from copy import deepcopy
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from icl.retrieval import RandomRetriever, retrieve_mention_classification_exemplars

from typing import Literal, Optional, Dict, Any



class Opinion(BaseModel):
    answer: Literal["Insbruck", "Vienna", "Salzburg"] = Field(..., 
        description="One-word answer to the multiple-choice opinion question."
    )
    explanation: str = Field(...,
        description="1-2 sentences long explanation of your opinion."
    )

class HasAttribute(BaseModel):
    reasoning: str = Field(..., description="One or two short sentences documenting your classification reasoning")
    classification: Literal["yes", "no"] = Field(..., description="Classification decision")

class Exemplar(BaseModel):
    # reasoning: str = Field(..., description="One or two short sentences documenting your classification reasoning")
    classification: Literal["yes", "no"] = Field(..., description="Classification decision")


class MentionAttributeClassificationProgram(LLMTextCompletionProgram):
    template = PromptTemplate(ATTRIBUTE_CLASSIFICATION_TASK_PROMPT_TEMPLATE)
    output_cls = HasAttribute
    # template_vars = ["text", "mention"]
    template_vars = ["mention"]
    max_tokens = 1024
    
    def __init__(
            self, 
            llm: LLM,
            attribute_id: str,
            attribute_name: str,
            attribute_definition: str,
            n_exemplars: Optional[int]=None,
            retriever=None,
            exemplars_sampling_strategy: Literal['random', 'similarity']='similarity',
            exemplars_reshuffle: bool=True,
            verbose: bool=False,
        ):
        """
        Args:
            llm (LLM): The LLM to use for text completion.
            attribute_id (str): The attribute identifier for filtering exemplars.
            attribute_name (str): Human-readable name of the attribute.
            attribute_definition (str): Definition of the attribute for the prompt.
            n_exemplars (int, optional): 
                Number of exemplars to use for few-shot classification. If None, zero-shot classification is used.
            retriever (optional):
                Required if n_exemplars > 0.
                A pre    Re retriever containing exemplars for all attributes with labels stored in metadata.
            exemplars_sampling_strategy (str): 
                Strategy for retrieving exemplars. Must be one of ['random', 'similarity'].
            verbose (bool): Whether to print verbose output.
                
        """
        self.attribute_id = attribute_id
        self.attribute_name = attribute_name
        self.attribute_definition = attribute_definition

        self.template = self.template.partial_format(
            attribute_name=self.attribute_name,
            attribute_definition=self.attribute_definition
        )

        # instantiate the LLMTextCompletionProgram
        super().__init__(
            llm=llm, 
            prompt=self.template, 
            output_cls=self.output_cls, 
            output_parser=PydanticOutputParser(output_cls=self.output_cls),
            verbose=verbose
        )

        # create the response format
        # TODO: consider whether this only applies to HF (inference) LLMs
        self.response_format = get_response_format(self.output_cls)

        self.llm_kwargs = None
        if isinstance(self._llm, OpenAILike):
            self.llm_kwargs = {"extra_body": {"response_format": _type_to_response_format(self.output_cls)}}

        # setup few-shot logic
        self.is_fewshot = False
        self.k_shots = False
        if n_exemplars:
            if n_exemplars <= 0:
                raise ValueError("n_exemplars must be > 0 for few-shot classification or `None` (zero-shot)")
            if retriever is None:
                raise ValueError("retriever must be provided if n_exemplars > 0")
            if exemplars_sampling_strategy not in ['random', 'similarity']:
                raise ValueError(f"exemplars_sampling_strategy must be one of ['random', 'similarity'], but got {exemplars_sampling_strategy}") 
            
            self.is_fewshot = True 
            self.k_shots = n_exemplars
            self.exemplars_sampling_strategy = exemplars_sampling_strategy
            self.exemplars_reshuffle = exemplars_reshuffle
            self.retriever = retriever
            
            # add an exemplars placeholder to the prompt template
            prompt_text = deepcopy(self.template.template)
            parts = prompt_text.split('\n## Input')
            input_format = None
            if len(parts)>1:
                prompt_text, input_format = parts[0].rstrip(), '\n## Input'+parts[1]
            prompt_text += '\n\n## Examples\n\n{exemplars}'
            if input_format:
                prompt_text += input_format
            # update the prompt template
            self.template = PromptTemplate(prompt_text)
            self.template = self.template.partial_format(
                attribute_name=attribute_name,
                attribute_definition=attribute_definition
            )
            self.prompt = deepcopy(self.template)
            self.prompt.function_mappings = {"exemplars": self._retrieve_mention_classification_exemplars}

    
    # # since we use huggingface inference providers for LLM inference, rely on support async calls
    # # NOTE: not sure about subclassing here
    # async def acall(
    #         self, 
    #         llm_kwargs: Optional[Dict[str, Any]] = None,
    #         *args: Any,
    #         **kwargs: Any,
    #     ) -> HasAttribute:
    #     llm_kwargs = {"response_format": self.response_format, "max_tokens": self.max_tokens, **(llm_kwargs or {})}
    #     return await self.super().acall(llm_kwargs=llm_kwargs, *args, **kwargs)


    def _retrieve_mention_classification_exemplars(self, mention, verbose: bool=False, **kwargs):
        if not self.is_fewshot:
            raise ValueError("Few-shot exemplars not set up")
        res = retrieve_mention_classification_exemplars(
            retriever=self.retriever, 
            k=self.k_shots,
            strategy=self.exemplars_sampling_strategy,
            mention=mention, # <== input text used for retrieval passed to kwargs of ``retrieve_mention_classification_exemplars()``
            text_field_name='text',
            mention_field_name='mention',
            # format_metadata_fun=lambda text, metadata: self.output_cls(reasoning='...', classification=metadata.get('classification', metadata.get(self.attribute_id))),
            format_metadata_fun=lambda text, metadata: Exemplar(classification=metadata.get('classification', metadata.get(self.attribute_id))),
            return_as='text',
            attribute_id=self.attribute_id,  # Pass attribute_id for filtering
            reshuffle=self.exemplars_reshuffle
        )
        if verbose:
            print( '--- retrieved exemplars ---')
            print(res)
            print( '---------------------------')
        
        return res
