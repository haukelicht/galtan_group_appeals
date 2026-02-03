from llama_index.core.schema import TextNode
from dataclasses import dataclass
from typing import List
import random
import json

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


def retrieve_mention_classification_exemplars(
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
    attribute_id: str=None,
    balance_labels: bool=True,
    reshuffle: bool=False,
    **kwargs
):
    """
    Function for selecting exemplars for a given sentence from retriever.
    
    Args:
        attribute_id (str, optional): If provided, filter retrieved nodes to only include 
            those with labels for this specific attribute in their metadata.
        balance_labels (bool): If True and attribute_id is provided, attempt to balance 
            positive and negative examples.
    """
    retrieved_nodes = retriever.retrieve(kwargs[mention_field_name])
    
    if strategy == 'similarity':
        # NOTE: similarity-based retriever returns a list of NodeWithScore objects that allow ranking
        # NOTE: we rank in ascending order so that more similar exemplars are later in prompt and thus closer to the to-be-labeled input (bc recency bias)
        retrieved_nodes = list(sorted(retrieved_nodes, key=lambda x: x.score, reverse=descending))
        
        # Filter by attribute_id if provided (for shared retriever with all attributes)
        if attribute_id is not None:
            # Separate positive and negative examples
            pos_nodes = [n for n in retrieved_nodes if n.metadata.get(attribute_id) == 'yes']
            neg_nodes = [n for n in retrieved_nodes if n.metadata.get(attribute_id) == 'no']
            
            if balance_labels and len(pos_nodes) > 0 and len(neg_nodes) > 0:
                # Try to balance: take k//2 from each, or as many as available
                k_pos = min(k // 2, len(pos_nodes))
                k_neg = min(k // 2, len(neg_nodes))
                
                # If we got fewer than k total, fill remaining slots from whichever class has more
                total = k_pos + k_neg
                if total < k:
                    remaining = k - total
                    if len(pos_nodes) > k_pos:
                        additional_pos = min(remaining, len(pos_nodes) - k_pos)
                        k_pos += additional_pos
                        remaining -= additional_pos
                    if remaining > 0 and len(neg_nodes) > k_neg:
                        k_neg += min(remaining, len(neg_nodes) - k_neg)
                
                retrieved_nodes = pos_nodes[:k_pos] + neg_nodes[:k_neg]
            else:
                # Just take top k from all examples with this attribute
                retrieved_nodes = (pos_nodes + neg_nodes)[:k]
        else:
            # No filtering, just take top k
            retrieved_nodes = retrieved_nodes[:k]
    
    exs = []
    for node in retrieved_nodes:
        mention = node.get_text()
        ex = {
            #text_field_name: node.metadata.get(text_field_name, '...'),
            mention_field_name: mention
        }
        annotations = format_metadata_fun(mention, node.metadata)
        ex['annotations'] = json.dumps(dict(annotations)) if return_as=='text' else annotations
        exs.append(ex)
    
    if reshuffle:
        random.Random(42).shuffle(exs)

    if return_as=='list':
        return exs
    
    # # text format for each input--output pair
    # fmt = '%s:\n%s: """{%s}"""\n%s: """{%s}"""\n%s: {annotations}' % (
    #     input_key_name, 
    #     text_field_name.title(), text_field_name,
    #     mention_field_name.title(), mention_field_name,
    #     response_key_name
    # )
    # text format for each input--output pair
    fmt = '%s:\n%s: """{%s}"""\n%s: {annotations}' % (
        input_key_name, 
        mention_field_name.title(), mention_field_name,
        response_key_name
    )
    # join all examples
    exs = '\n\n'.join([fmt.format(**ex) for ex in exs])
    
    return exs
