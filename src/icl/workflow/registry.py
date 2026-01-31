from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Union, Generator
import json

@dataclass
class MentionCandidateTracker:
    id: int
    mention: str
    span: Optional[Tuple[int, int]] = None
    located: bool = False
    review_attempts: int = 0
    correction_attempts: int = 0
    validation_attempts: int = 0    
    valid: Union[None, bool] = None
    discarded: bool = False
    discard_reason: Optional[str] = None
    
    def __post_init__(self):
        if self.span is not None:
            self.located = True

    def set_span(self, span: Tuple[int, int]):
        """ Set span when it is determined later in the process. """
        if self.span is None:
            self.span = span
            self.located = True
    
    def discard(self, reason: str):
        """ Discard a mention candidate with a reason for discarding. """
        if not self.discarded:
            self.discarded = True
        self.discard_reason = reason

    def __str__(self):
        return (f'MentionCandidateTracker(id={self.id}, mention="{self.mention}", '
                f'span={self.span}, located={self.located}, valid={self.valid}, '
                f'discarded={self.discarded}, discard_reason="{self.discard_reason}"'
                ')'
               )
    
    def to_dict(self):
        return self.__dict__
    
    def to_json(self):
        """For serialization to JSON"""
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False)
    

@dataclass
class WorkflowRegistry:
    """ Registry for tracking mention candidates in a text.
    
    This class allows adding, updating, and retrieving mention candidates,
    along with their spans and validation status. It supports adding multiple
    mentions at once, updating spans, and iterating over valid candidates.

    Attributes:
        text (str): The text in which mentions are tracked.
        n_candidates (int): The number of candidates added so far.
        mentions (List[str]): List of initial mentions to track.
        candidates (OrderedDict): Dictionary of mention candidates indexed by their IDs.
        mention_to_ids (Dict[str, list]): Maps each mention to a list of candidate IDs.

    """


    text: str
    n_candidates: int = 0
    mentions: List[str] = field(default_factory=list)
    candidates: OrderedDict = field(default_factory=OrderedDict)
    mention_to_ids: Dict[str, list] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.mentions) > 0:
            self.add_mentions(self.mentions, [None]*len(self.mentions))
        
    def new_id(self):
        self.n_candidates += 1
        return self.n_candidates
    
    def add_candidate(
            self, 
            mention: str, 
            span: Optional[Tuple[int, int]] = None,
            return_candidate: bool = False
        ):
        """ Adds a new mention candidate, optionally setting the span if known. """
        id = self.new_id()
        self.candidates[id] = MentionCandidateTracker(id=id, mention=mention, span=span)
        self.mention_to_ids.setdefault(mention, []).append(id)
        return self.candidates[id] if return_candidate else id
    
    def add_mentions(self, mentions: List[str], spans: Optional[List[Tuple[int, int]]] = None):
        """ Adds multiple mention candidates, optionally setting spans if known. """
        for mention, span in zip(mentions, spans or [None]*len(mentions)):
            self.add_candidate(mention, span)

    def get_candidate(self, id: int):
        return self.candidates[id]
    
    def update_candidate(self, candidate: MentionCandidateTracker):
        self.candidates[candidate.id] = candidate
    
    def update_candidate_span(self, mention: str, span: Tuple[int, int]):
        """ Updates the span of an existing mention if found later in processing. """
        for candidate_id in self.find_candidates(mention):
            candidate = self.get_candidate(candidate_id)
            if candidate.span is None:  # Only update if span was not already set
                candidate.set_span(span)
                return candidate_id  # Return the updated candidate's ID
        return None  # No matching mention found to update

    def __iter__(self):
        """ Iterate over valid, non-discarded mentions. """
        for _, c in self.candidates.items():
            if not c.discarded:
                yield c
    
    @property
    def valid_candidates(self) -> Generator[MentionCandidateTracker, None, None]:
        """ Iterate over valid, non-discarded mentions. """
        for _, c in self.candidates.items():
            if not c.discarded and c.valid is not None and c.valid:
                yield c

    @property
    def valid_mentions(self) -> Generator[str, None, None]:
        for c in self.valid_candidates:
            yield c.mention

    def find_candidates(self, mention):
        """ Return all candidate IDs for a given mention. """
        return self.mention_to_ids.get(mention, [])
    
    def __contains__(self, mention):
        """ Checks if a mention exists in any form. """
        return mention in self.mention_to_ids
    
    def __str__(self):
        return (
            'WorkflowRegistry(\n  '
            f'text: "{self.text}"\n  '
            f'candidates:\n    ' 
            '%s'
            '\n)'
        ) % '\n    '.join([str(c) for c in self.candidates.values()])

    def __repr__(self):
        return str(self)
    
    def __len__(self):
        return len(self.candidates)
    
    def to_json(self):
        """For serializing the registry to JSON."""
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False)
    
    @classmethod
    def from_json(cls, x: Union[str, Dict]) -> 'WorkflowRegistry':
        """Create a WorkflowRegistry from a JSON string."""
        if isinstance(x, str):
            x = json.loads(x)
        registry = cls(text=x['text'], n_candidates=x['n_candidates'])
        for mention, ids in x['mention_to_ids'].items():
            registry.mention_to_ids[mention] = ids
        for candidate in x['candidates'].values():
            registry.update_candidate(MentionCandidateTracker(**candidate))
        return registry