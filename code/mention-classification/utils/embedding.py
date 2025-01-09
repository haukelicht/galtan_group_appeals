import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import gc
import math

from tqdm.auto import tqdm

from dataclasses import dataclass
from typing import List, Literal, Union

def clean_memory(device: Union[str, torch.device]):
    gc.collect()
    if str(device) == 'cuda':
        torch.cuda.empty_cache()
    elif str(device) == 'mps':
        torch.mps.empty_cache()
    else:
        pass

@dataclass
class E5SentenceEmbedder:
    model_name: str = 'intfloat/multilingual-e5-base'
    device: Literal['cuda', 'mps', 'cpu'] = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device);

    @staticmethod
    def _average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Source: https://huggingface.co/intfloat/multilingual-e5-base
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def encode(self, texts: List[str], batch_size: int=16, normalize: bool=True) -> torch.Tensor:
        """
        Source: based on https://huggingface.co/intfloat/multilingual-e5-base
        """
        # Each input text should start with "passage: ", even for non-English texts.
        # For tasks other than retrieval, you can simply use the "query: " prefix.
        texts = ['query: ' + text if not text.lower().startswith('query: ') else text for text in texts]

        embeddings = []
        n_ = len(texts)
        for i in tqdm(range(0, n_, batch_size), total=math.ceil(n_/batch_size)):
            batch_dict = self.tokenizer(texts[i:min(i+batch_size, n_)], max_length=512, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**batch_dict.to(self.model.device))
            tmp = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).half().cpu()
            embeddings.append(tmp)
            del tmp
            clean_memory(str(self.device))
        embeddings = torch.cat(embeddings, dim=0)
        
        # normalize embeddings
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings