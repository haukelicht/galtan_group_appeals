
import torch

from setfit import SetFitHead, SetFitModel
from sentence_transformers import SentenceTransformer

from .class_weights_head import SetFitHeadWithClassWeights
from .span_embedding import SentenceTransformerForSpanEmbedding, SetFitModelForSpanClassification

from typing import Mapping, Optional, Union
from numpy._typing import NDArray

get_device = lambda: 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def model_init(
        model_name: str,
        id2label: Mapping[int, str],
        multitarget_strategy: Optional[str]=None,
        use_span_embedding: bool=True,
        class_weights: Optional[NDArray]=None,
        device: Optional[Union[str, torch.device]]=None
    ) -> "SetFitModel":
    if class_weights is not None:
      if multitarget_strategy is None:
        if len(id2label) != len(class_weights):
            raise ValueError('length of `class_weights` must be same as `out_features`')

    if device is None:
        device = get_device()

    body = SentenceTransformerForSpanEmbedding(model_name, device='cpu') if use_span_embedding else SentenceTransformer(model_name, device='cpu')

    head_kwargs = dict(
        in_features=body.get_sentence_embedding_dimension(),
        out_features=len(id2label),
        device='cpu',
        multitarget=isinstance(multitarget_strategy, str),
    )
    if class_weights is not None:
        head_kwargs['class_weights'] = class_weights
        head = SetFitHeadWithClassWeights(**head_kwargs)
    else:
        head = SetFitHead(**head_kwargs)

    ModelClass = SetFitModelForSpanClassification if use_span_embedding else SetFitModel
    return ModelClass(
        model_head=head,
        model_body=body,
        multitarget_strategy=multitarget_strategy,
        labels=list(id2label.values()),
        id2label=id2label
    ).to(device)
