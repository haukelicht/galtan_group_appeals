"""
# Usage:

from llama_index.llms.ollama import Ollama
from icl.workflow.programs import MentionTypeValidator
from icl.workflow.registry import WorkflowRegistry

llm = Ollama(model='phi4', temperature=0.0, seed=42, json_mode=True)
mention_type_validator = MentionTypeValidator(llm=llm, verbose=True)
w = MentionsTypeClassificationWorkflow(validator=mention_type_validator, verbose=True)

sentence = "This has meant that landlords and housing companies increased rents for the better off tenants who had hitherto often been enjoying bigger subsidies than many poorer families."
candidates = ['landlords', 'housing companies', 'better off tenants', 'poorer families']

reg = WorkflowRegistry(text=sentence, mentions=candidates)
await w.run(registry=reg)
"""

# from utils.log import log, ts
import warnings
format_warning = lambda message, category, filename, lineno, line=None: f"[{ts()}]: {category.__name__}: {message}\n"
warnings.formatwarning = format_warning

from llama_index.core.workflow import (
    Workflow,
    step,
    StartEvent,
    StopEvent,
    Event,
    Context
)

from typing import List, Any
# from icl.workflow.registry import WorkflowRegistry, MentionCandidateTracker
from icl.attribute_classification_program import HasAttribute, AttributeClassificationProgram

from icl.definitions import attribute_names, attribute_definitions

class ClassificationEvent(Event):
    candidate: MentionCandidateTracker

class ClassificationResult(Event):
    candidate: MentionCandidateTracker

@dataclass
class ClassificationOutput:
    classifications: Dict[str, Dict[str, Any]]  # attribute_name -> {classification: str,
    reasoning: str

class AttributesClassificationWorkflow(Workflow):

    def __init__(
            self, 
            *args,
            verbose: bool=False,
            **kwargs: Any, 
        ):
        super().__init__(*args, **kwargs)
        # NOTE: we overwrite the verbose flag to avoid verbose output from the workflow
        self.verbose = verbose

    def run(self, text: str, mention: str) -> WorkflowRegistry:
        return super().run(registry=registry)
    
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> ClassificationEvent | ClassificationResult:
        # get non-discarded candidates
        candidates = [c for c in ev.registry]
        if len(candidates)==0:
            return StopEvent(result=ev.registry) 
        await ctx.set("registry", ev.registry)
        await ctx.set("text", ev.registry.text)
        await ctx.set("n_candidates", len(candidates))
        if self.verbose: log(f'Validating group type of {len(candidates)} mention candidates.')
        for c in candidates:
            if c.valid is None and not c.discarded:
                ctx.send_event(ClassificationEvent(candidate=c))
            else:
                if self.verbose: log(f'Skipping mention "{c.mention}" because already validated')
                ctx.send_event(ClassificationResult(candidate=c))

    @step(num_workers=10)
    async def validate_group_type(self, ctx: Context, ev: ClassificationEvent) -> ClassificationResult:
        """Given input text and a consolidated extraction, assistant double-checks if the extracted mention refers to a social group and not some other type of group, collective, or entity"""
        if self.verbose: log(f'Validating type of mention "{ev.candidate.mention}"')
        text = await ctx.get("text")
        output = self.validator(text=text, mention=ev.candidate.mention)
        candidate = ev.candidate
        candidate.validation_attempts += 1
        if output.response=='yes':
            if self.verbose: log(f'Validated type of mention "{candidate.mention}": valid')
            candidate.valid = True
        # elif output.response=='no' and ev.consolidate_if_needed:
        #     return ConsolidateMentionCandidateEvent(mention=ev.mention, order=ev.order, start=ev.start, end=ev.end)
        elif output.response=='no':
            if self.verbose: log(f'Validated type of mention "{candidate.mention}": invalid')
            candidate.valid = False
            candidate.discard(f'Discarded due to invalid type. Reason: "{output.reasoning}"')
        else:
            raise ValueError()
        return ClassificationResult(candidate=candidate)
    
    # @step(num_workers=10)
    # async def consolidate_candidate_mention(self, ctx: Context, ev: ConsolidationEvent) -> ClassificationEvent:
    #     if self.verbose: log(f'Consolidating extent of mention candidate "{ev.mention}"')
    #     text = await ctx.get("text")
    #     output = self.consolidator(text=text, mention=ev.candidate.mention)
    #     # if output.mention!=ev.mention:
    #     #     edits = await ctx.get("candidate_mentions_edits", [])
    #     #     edits.append(f"consolidation: Correct candidate mention from '{ev.mention}' to '{output.mention}'")
    #     #     await ctx.set("candidate_mentions_edits", edits)
    #     # validate type again but without another follow-up condsolidation step
    #     return ClassificationEvent(ev.candidate)

    @step
    async def collect_results(self, ctx: Context, ev: ClassificationResult) -> StopEvent:
        n = await ctx.get("n_candidates")
        if self.verbose: log(f'Collecting {n} validation results')
        results = ctx.collect_events(ev, [ClassificationResult] * n)
        if results is not None:
            if self.verbose: log(f'Collected {n} validation results')
            registry = await ctx.get("registry")
            for res in results:
                registry.update_candidate(res.candidate)
            return StopEvent(result=registry)
