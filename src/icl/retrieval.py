from llama_index.core.schema import TextNode
from dataclasses import dataclass
from typing import List
import random

@dataclass
class RandomRetriever:
    exemplars: List[TextNode]
    k: int
    seed: int = 42

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def retrieve(self, sentence=None):
        return self.rng.sample(self.exemplars, self.k)

from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from typing import Union, Literal, Callable

def retrieve_exemplars(
    retriever: Union[RandomRetriever, VectorIndexRetriever],
    k: int,
    format_metadata_fun: Callable,
    text_field_name: str='sentence',
    strategy: Literal['random', 'similarity']='random',
    input_key_name: str='INPUT',
    response_key_name: str='RESPONSE',
    return_as: Literal['list', 'text']='text',
    descending: bool=False,
    **kwargs
):
    """Function for selecting exemplars for a given sentence from retriever"""
    retrieved_nodes = retriever.retrieve(kwargs[text_field_name])
    if strategy == 'similarity':
        # NOTE: similarity-based retriever returns a list of NodeWithScore objects that allow ranking
        # NOTE: we rank in ascending order so that more similar exemplars are later in prompt and thus closer to the to-be-labeled input (bc recency bias)
        retrieved_nodes = list(sorted(retrieved_nodes, key=lambda x: x.score, reverse=descending)[-k:])
    exs = []
    for node in retrieved_nodes:
        text = node.get_text()
        ex = {text_field_name: text}
        annotations = format_metadata_fun(text, node.metadata)
        ex['annotations'] = annotations.model_dump(mode='json') if return_as=='text' else annotations
        exs.append(ex)
    
    if return_as=='list':
        return exs
    
    # text format for each input--output pair
    fmt = '%s: """{%s}"""\n%s: {annotations}' % (input_key_name, text_field_name, response_key_name)
    # join all examples
    exs = '\n\n'.join([fmt.format(**ex) for ex in exs])
    
    return exs

def retrieve_mention_exemplars(
    retriever: Union[RandomRetriever, VectorIndexRetriever],
    k: int,
    format_metadata_fun: Callable,
    text_field_name: str='text',
    mention_field_name: str='mention',
    strategy: Literal['random', 'similarity']='random',
    input_key_name: str='INPUT',
    response_key_name: str='RESPONSE',
    return_as: Literal['list', 'text']='text',
    descending: bool=False,
    **kwargs
):
    """Function for selecting exemplars for a given sentence from retriever"""
    retrieved_nodes = retriever.retrieve(kwargs[mention_field_name])
    if strategy == 'similarity':
        # NOTE: similarity-based retriever returns a list of NodeWithScore objects that allow ranking
        # NOTE: we rank in ascending order so that more similar exemplars are later in prompt and thus closer to the to-be-labeled input (bc recency bias)
        retrieved_nodes = list(sorted(retrieved_nodes, key=lambda x: x.score, reverse=descending)[-k:])
    exs = []
    for node in retrieved_nodes:
        mention = node.get_text()
        ex = {
            text_field_name: node.metadata.get(text_field_name, '...'),
            mention_field_name: mention
        }
        annotations = format_metadata_fun(mention, node.metadata)
        ex['annotations'] = annotations.model_dump(mode='json') if return_as=='text' else annotations
        exs.append(ex)
    
    if return_as=='list':
        return exs
    
    # text format for each input--output pair
    fmt = '%s:\nSentence: """{%s}"""\nMention: """{%s}"""\n%s: {annotations}' % (input_key_name, text_field_name, mention_field_name, response_key_name)
    # join all examples
    exs = '\n\n'.join([fmt.format(**ex) for ex in exs])
    
    return exs
