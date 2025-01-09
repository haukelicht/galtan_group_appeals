import json

import re
import numpy as np
from itertools import chain
from nltk import word_tokenize

from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Protocol

# utils
def format_doccano_annotations(annotations: List[List]) -> List[Dict]:
    out = annotations.copy()
    n = len(annotations)
    for i in range(n):
        s, e, c = annotations[i]
        out[i]= {'span': [s, e], 'type': c}
    return out

def locate_mentions_in_text(
        text: str, 
        mentions: List[str],
        categories: Optional[List[str]] = None
    ) -> Dict:
    """
    Locate mentions in the text and return their spans.
    """
    if categories is None:
        categories = [None] * len(mentions)
    
    located = []
    for m, c in zip(mentions, categories):
        none = True
        for span in re.finditer(m, text):
            none = False
            located.append((span.start(), span.end(), c))
        if none: 
            raise ValueError(f'Mention "{m}" not found in the text.')

    # make unique
    located = list(set(located))
    
    # sort by start index
    located = sorted(located, key=lambda x: x[0])

    located = [{'span': [s, e], 'category': c} for s, e, c in located]

    return located

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

@dataclass
class Text:
    id: Union[str, int]
    text: str

    @staticmethod
    def from_dict(d) -> 'Text':
        return Text(id=d['id'], text=d['text'])

    def __str__(self):
        return f'\x1b[1m{self.id}\x1b[0m: "{self.text}"'

    def __len__(self) -> int:
        return len(self.text)
    
    # def __dict__(self) -> dict:
    #     return self.__dict__()
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__(), cls=NpEncoder)


@dataclass
class Span:
    start: int
    end: int
    mention: str

    def __len__(self) -> int:
        return self.end - self.start

    def __dict__(self) -> Dict:
        return {'start': self.start, 'end': self.end, 'mention': self.mention}
    
@dataclass
class Entity(Span):
    type: str

    def __dict__(self) -> Dict:
        return {'start': self.start, 'end': self.end, 'mention': self.mention, 'type': self.type}

@dataclass
class LabeledText:
    """Class representing a text annotated with spans containg entity mentions."""
    id: Union[str, int]
    text: str
    entities: List[Entity]
    lang: Optional[str] = 'english'
    # TODO: handle any text-level metadata
    
    @staticmethod
    def from_dict(d: Dict, entities_field: str='annotations', type_field: Optional[str]='type') -> 'LabeledText':
        assert 'id' in d, '"id" field missing in dictionary'
        assert 'text' in d, '"text" field missing in dictionary'
        assert entities_field in d, f'"{entities_field}" field missing in dictionary'
        assert isinstance(d[entities_field], List), f'"{entities_field}" field must be a list'
        if len(d[entities_field]) > 0:
            assert all(isinstance(e, Dict) and 'span' in e for e in d[entities_field]), f'all entries in "{entities_field}" field must be dictionaries with field "span"'
            # assert all(type_field in e for e in d[entities_field]), f'all entries in "{entities_field}" field must be dictionaries with field "{type_field}"'
        entities = []
        for entity in d[entities_field]:
            #s, e = entity[0], entity[1]
            s, e = entity['span']
            m = d['text'][s:e]
            entity = Entity(start=s, end=e, mention=m, type=entity[type_field] if type_field in entity else None)
            entities.append(entity)
        entities = sorted(entities, key=lambda x: x.start)
        return LabeledText(id=d['id'], text=d['text'], entities=entities)
    
    def _generate_printable(self):
        if len(self.entities) == 0:
            self._printable = f'\x1b[1m{self.id}\x1b[0m: "{self.text}"'
            return self._printable
        
        # highlight annotations
        e = self.entities[0].start
        text = self.text[:e]
        for span in self.entities:
            text += self.text[e:span.start]
            text += f'\x1b[43m\x1b[1m{self.text[span.start:span.end]}\x1b[0m\x1b[43m'
            if span.type:
                text += f' [{span.type}]'
            text += '\x1b[49m'
            e = span.end
        if len(self.text) > e:
            text += self.text[e:]
        self._printable = f'\x1b[1m{self.id}\x1b[0m: "{text}"'
    
    def __str__(self):
        if not hasattr(self, '_printable'):
            self._generate_printable()
        return self._printable

    def __getitem__(self, idx: int) -> Entity:
        return self.entities[idx]
    
    def __len__(self) -> int:
        return len(self.entities)
    
    def __dict__(self) -> Dict:
        return {'id': self.id, 'text': self.text, 'entities': [entity.__dict__() for entity in self.entities]}
    
    def to_dict(self) -> Dict:
        return self.__dict__()
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__(), cls=NpEncoder)

# word tokenization helper
def word_tokenize_with_whitespace(text: str, lang: str):
    # split into tokens, preseving whitespaces
    toks = [[word_tokenize(tok, language=lang), ' '] for tok in re.split(r'\s', text)]
    toks = list(chain(*list(chain(*toks))))
    if toks[-1] == ' ':
        toks = toks[:-1]
    return toks

class TokenizerFunctionTemplate(Protocol):
    def __call__(self, text: str, lang: str) -> List[str]:
        pass

@dataclass
class LabeledSequence:
    """Class representing a text labeled with token-level entity mention annotations."""
    id: Union[str, int]
    text: str
    tokens: List[str]
    annotations: List[int]
    label2id: Dict[str, int]
    id2label: Optional[Dict[int, str]] = None
    lang: Optional[str] = 'english'

    def __post_init__(self):
        if self.id2label is None:
            self.id2label = {i: l for l, i in self.label2id.items()}

    @staticmethod
    def from_labeled_text(doc: LabeledText, label2id: Dict[str, int] = None, tokenizer: TokenizerFunctionTemplate=word_tokenize_with_whitespace) -> 'LabeledSequence':
        
        # split into tokens, preseving whitespaces
        toks = tokenizer(text=doc.text, lang=doc.lang)

        # get token indexes
        word_indexes = []
        for i, tok in enumerate(toks):
            tmp = list(tok)
            word_indexes += [-1] * len(tmp) if re.match(r'\s', tok) else [i] * len(tmp)

        # transfer annotaions to tokens
        aidxs = np.zeros(len(toks), dtype=np.int16)
        for entity in doc.entities:
            idxs = word_indexes[entity.start:entity.end]
            idxs = np.array(idxs)
            # get all values > -1
            widxs = sorted(list(set(idxs[idxs > -1])))
            i = widxs.pop(0)
            aidxs[i] = label2id['B-'+entity.type]
            aidxs[widxs] = label2id['I-'+entity.type]

        # add as attributes
        annotations = [int(a) for t, a in zip(toks, aidxs) if not re.match(r'\s+', t)]
        tokens = []
        for i, tok in enumerate(toks):
            prefix = '' if i == 0 else ' ' if re.match(r'\s+', toks[i-1]) else ''
            if not re.match(r'\s+', tok):
                tokens.append(prefix+tok)

        return LabeledSequence(id=doc.id, text=doc.text, tokens=tokens, annotations=annotations, label2id=label2id)

    def to_labeled_text(self):
        s = 0
        spans = []
        inside = False
        for t, a in self:
            e = s + len(t)
            if a in self.id2label and self.id2label[a] != 'O':
                if not inside:
                    s_ = s+1 if re.match(r'\s', t[0]) else s
                    spans.append({'span': [s_, e], 'category': self.id2label[a]})
                    inside = True
                else:
                    spans[-1]['span'][1] = e
            else:
                inside = False
            s += len(t)
        return LabeledText.from_dict({'id': self.id, 'text': self.text, 'annotations': spans, 'lang': self.lang})
    
    def __getitem__(self, idx: int) -> str:
        return self.tokens[idx], self.annotations[idx]
    
    def __len__(self) -> int:
        return len(self.tokens)
    
    def __iter__(self):
        return iter(zip(self.tokens, self.annotations))
    
    def __dict__(self) -> Dict:
        return {'id': self.id, 'text': self.text, 'tokens': self.tokens, 'annotations': self.annotations}
    
    def to_dict(self) -> Dict:
        return self.__dict__()
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__(), cls=NpEncoder)

