# coding=utf-8
# - USAGE -------------------------------
# :
#   from icl.workflows.mention_attributes_classifier import GroupMentionAttributesClassifierWorkflow
#   attributes_definitions = {
#       "economic__class_membership": {"name": "class membership", "definition":  "People described with their membership in or belonging to a social class"}
#       "economic__occupation_profession": {"name": "occupation/profession", "definition":  "People referred to with or categorized according to their occupation or profession"}
#       ...
#   }
#   attributes = Attributes.from_dict(attributes_definitions)
#   llm = ...  # initialize your LLM here
#   workflow = GroupMentionAttributesClassifierWorkflow(
#       llm=llm,
#       attributes=attributes,
#       verbose=True,
#   )
#   text, mention = "The input text that contains the to-be-classified mention.", "the to-be-classified mention"
#   output = await workflow.run(text=text, mention=mention)
#   output.to_pandas()
# 
# ----------------------------------------

from icl._utils.log import log, ts
import warnings
format_warning = lambda message, category, filename, lineno, line=None: f"[{ts()}]: {category.__name__}: {message}\n"
warnings.formatwarning = format_warning

import pandas as pd

from llama_index.core.llms.llm import LLM
from icl.definitions import Attributes
from icl.program import (
    HasAttribute,
    MentionAttributeClassificationProgram
)
from llama_index.core.workflow import (
    Workflow,
    step,
    StartEvent,
    StopEvent,
    Event,
    Context
)

# Build a single index with ALL attribute labels in metadata
from pathlib import Path
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from icl.retrieval import RandomRetriever

from typing import List, Literal, Optional, Dict, Any
from dataclasses import dataclass

class ClassificationEvent(Event):
    text: str
    mention: str
    attribute_id: str

class ClassificationResult(Event):
    attribute_id: str
    result: HasAttribute

@dataclass
class ClassificationsResult:
    text: str
    mention: str
    classifications: Optional[Dict[str, HasAttribute]] = None
    mention_kwargs: Dict[str, Any] = None

    @property
    def labels(self) -> Dict[str, str]:
        return {
            k: v.classification
            for k, v in self.classifications.items()
        }
        
    def to_pandas(self) -> pd.DataFrame:
        meta = self.__dict__.copy()
        mention_kwargs = meta.pop('mention_kwargs')
        if mention_kwargs is None:
            mention_kwargs = {}
        classifications = meta.pop('classifications')
        n_rows = len(classifications) if classifications else 1
        meta = pd.DataFrame([{**meta, **mention_kwargs}]*n_rows)
        if classifications is None or len(classifications) == 0:
            return meta
        classifications = pd.DataFrame.from_dict({a: dict(d) if d is not None else {} for a, d in classifications.items()}, orient='index').reset_index(names='attribute_id')
        return pd.concat([meta, classifications], axis=1)


class GroupMentionAttributesClassifierWorkflow(Workflow):
    
    def __init__(
            self, 
            *args,
            llm: LLM,
            attributes: Attributes,
            exemplars_data: pd.DataFrame=None,
            exemplars_file: str=None,
            n_exemplars: Optional[int]=0,
            exemplars_sampling_strategy: Literal['random', 'similarity']='similarity',
            exemplars_embedding_model: str=None,
            verbose: bool=False,
            **kwargs: Any, 
        ):
        super().__init__(*args, **kwargs)
        self.attributes = attributes
        
        # Setup shared retriever for few-shot learning
        retriever = None
        if n_exemplars:
            if n_exemplars <= 0:
                raise ValueError("n_exemplars must be > 0 for few-shot classification or `None` (zero-shot)")
            elif exemplars_data is None and exemplars_file is None:
                raise ValueError("Either exemplars_data or exemplars_file must be provided if n_exemplars > 0")
            elif exemplars_data is not None and exemplars_file is not None:
                raise ValueError("Only one of exemplars_data or exemplars_file must be provided if n_exemplars > 0")
            elif n_exemplars > 0:
                # Load exemplars data
                if exemplars_data is None:
                    exemplars_file = Path(exemplars_file).resolve()
                    if not exemplars_file.exists():
                        raise FileNotFoundError(f"Exemplars file {exemplars_file} not found")
                    # try loading the exemplars file
                    ext = exemplars_file.suffix.lstrip('.')
                    if ext == 'pkl':
                        exemplars_data = pd.read_pickle(exemplars_file)
                    elif ext in ('tsv', 'tab'):
                        exemplars_data = pd.read_csv(exemplars_file, sep='\t')
                    elif ext in ('csv'):
                        exemplars_data = pd.read_csv(exemplars_file)
                    elif ext == 'jsonl':
                        exemplars_data = pd.read_json(exemplars_file, lines=True)
                    else:
                        raise ValueError(f"Unsupported exemplars file format: {ext}")
                
                # ensure that all attributes are present in the columns
                missing_attributes = [a for a, _ in self.attributes if a not in exemplars_data.columns]
                if len(missing_attributes) > 0:
                    raise ValueError(f"Exemplars data is missing columns for attributes: {missing_attributes}")
                
                
                exemplar_nodes = []
                for _, row in exemplars_data.iterrows():
                    # Store ALL attribute labels in metadata
                    metadata = {'text': row['text']}
                    for a, _ in self.attributes:
                        metadata[a] = 'yes' if row[a] == 1 else 'no'
                    
                    exemplar_nodes.append(
                        TextNode(text=row['mention'], metadata=metadata)
                    )
                
                # Initialize the shared retriever
                if exemplars_sampling_strategy == 'random':
                    retriever = RandomRetriever(exemplar_nodes, k=n_exemplars)
                elif exemplars_sampling_strategy == 'similarity':
                    encoder = HuggingFaceEmbedding(exemplars_embedding_model)
                    # Retrieve more candidates than needed to allow for filtering
                    retriever = VectorStoreIndex(exemplar_nodes, embed_model=encoder).as_retriever(similarity_top_k=n_exemplars*5)
                    encoder._model.to('cpu')  # free up GPU memory
                    del encoder
                else:
                    raise ValueError(f"exemplars_sampling_strategy must be one of ['random', 'similarity'], but got {exemplars_sampling_strategy}")
        
        self.classifiers = {
            a: MentionAttributeClassificationProgram(
                llm=llm,
                attribute_id=a,
                attribute_name=d.name,
                attribute_definition=d.definition,
                n_exemplars=n_exemplars,
                retriever=retriever,
                exemplars_sampling_strategy=exemplars_sampling_strategy,
                verbose=verbose,
            )
            for a, d in self.attributes
        }

        # NOTE: we overwrite the verbose flag to avoid verbose output from the workflow
        self.verbose = verbose

    @property
    def n_attributes(self) -> int:
        return len(self.attributes)
    
    @property
    def attribute_ids(self) -> List[str]:
        return self.attributes.attribute_ids
    
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> ClassificationEvent:
        await ctx.store.set("text", ev.text)
        await ctx.store.set("mention", ev.mention)
        # TODO: get any other user-defined attributes

        if self.verbose: log(f'Classifying mention "{ev.mention}" into {len(self.attributes)} attribute categories.')
        for a in self.attribute_ids:
            ctx.send_event(ClassificationEvent(text=ev.text, mention=ev.mention, attribute_id=a))
            
    @step(num_workers=14) # = number of attributes to classify in parallel
    async def classify_one(self, ctx: Context, ev: ClassificationEvent) -> ClassificationResult:
        # if self.verbose: log(f'  - {self.attributes[ev.attribute_id].name}')
        output = await self.classifiers[ev.attribute_id].acall(text=ev.text, mention=ev.mention)
        if output.classification not in ('yes', 'no'):
            raise ValueError()
        return ClassificationResult(attribute_id=ev.attribute_id, result=output)
    
    @step
    async def collect_results(self, ctx: Context, ev: ClassificationResult) -> StopEvent:
        n = len(self.attributes)
        results = ctx.collect_events(ev, [ClassificationResult] * n)
        if results is None:
            return None
        
        if self.verbose: log(f'Collected {n} classification results')
        classifications = {
            res.attribute_id: res.result
            for res in results
        }
        return StopEvent(result=ClassificationsResult(
            text=await ctx.store.get("text"),
            mention=await ctx.store.get("mention"),
            mention_kwargs=await ctx.store.get("mention_kwargs", {}),
            classifications={a: classifications[a] for a in self.attribute_ids}
        ))
