import sys
import os
import asyncio
from tqdm import tqdm

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any

from llama_index.llms.ollama import Ollama
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.workflow import (
    Workflow,
    step,
    StartEvent,
    StopEvent,
    Event,
    Context
)

from utils.io import write_jsonlines

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('--llm', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, default=None)
    parser.add_argument('--overwrite_output_file', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--n_test', type=int, default=10)
    args = parser.parse_args()
else:
    from types import SimpleNamespace
    args = SimpleNamespace()
    args.input_file = './../../data/annotations/group_mention_categorization/social-group-mentions-pair-classification/sample.tsv'
    args.llm = 'phi4'
    args.output_file = None
    args.overwrite_output_file = True
    args.test = True

# ### Define workflow components, output schemata, and programs

# init LLM
llm = Ollama(
    model=args.llm,
    temperature=0.0,
    seed=42,
    json_mode=True,
    request_timeout=30.0
)

INPUT_VALIDATION_PROMPT = """\
Below you see a quote that should name, refer to, or describe a **social group mention**, that is, a collective of people that share some traits.\
These share traits might be demographic or socio-economic characteristics, related to group members' social or economic poistions, their profession or occupation, or their shared values, habits, or experiences.

Quote: \"\"\"{mention}\"\"\"\

Please indicate whether or not the quote qualifies as a social group mention according to the definition given above.
"""

YesNo = Literal["yes", "no"]
class MentionValidationOutput(BaseModel):
    valid: YesNo = Field(..., description='Choice indicating whether or not the quote qualifies as a "social group mention".')

class MentionValidationEvent(Event):
    pass # decision: MentionValidationOutput

validator = LLMTextCompletionProgram.from_defaults(
    output_cls=MentionValidationOutput,
    prompt_template_str=INPUT_VALIDATION_PROMPT,
    llm=llm,
    verbose=True,
)


MENTION_ANALYSIS_PROMPT = """\
Below you see a quote that names, refers to, or describes a social group (a collective of people with shared attributes):

Quote: \"\"\"{mention}\"\"\"\

Analyze what kind of people this quote refers to.\
Focus on the attributes and/or social categories that are used to delineate who belongs to the group.\
Respond with 1-2 sentences.
"""

class MentionAnalysisOutput(BaseModel):
    analysis: str = Field(..., description='Analysis summarizing what kind of people the quote refers to.')

class MentionAnalysisEvent(Event):
    pass # response: MentionAnalysisOutput

analyzer = LLMTextCompletionProgram.from_defaults(
    output_cls=MentionAnalysisOutput,
    prompt_template_str=MENTION_ANALYSIS_PROMPT,
    llm=llm,
    verbose=True,
)


PAIR_ANALYSIS_PROMPT = """\
Below you see two quotes that each name, refer to, or describe a social group (a collective of people with shared attributes)\
and 1-2 sentences that describe what kind of people they refer to:

- \"\"\"{mention_a}\"\"\": {description_a} 
- \"\"\"{mention_b}\"\"\": {description_b} 

Analyze whether or not these two quotes refers to the same group.\
Summarize your reasoning in 1-2 sentences.\
Then indicate your decision by replying either "yes" or "no" 
"""

class PairAnalysisOutput(BaseModel):
    reasoning: str = Field(..., description='Reasoning of analysis of whether or not two mentions refer to the same group of people.')
    decision: YesNo = Field(..., description='Decision indicating whether or not two mentions refer to the same group of people.')

class PairAnalysisEvent(Event):
    pass # response: PairAnalysisOutput

pair_analyzer = LLMTextCompletionProgram.from_defaults(
    output_cls=PairAnalysisOutput,
    prompt_template_str=PAIR_ANALYSIS_PROMPT,
    llm=llm,
    verbose=True,
)


# ### Define the workflow

class WorkflowOutput(BaseModel):
    inputs: List[str]
    analyses: List[str] | List[None] | None=None
    reasoning: str | None=None
    label: YesNo | None=None
    no_label_cause: str | None=None

class FinalEvent(Event):
    pass # label: Optional[YesNo]=None

class MentionPairClassificationWorkflow(Workflow):
    
    def __init__(
            self,
            *args: Any,
            mention_validator: "LLMTextCompletionProgram",
            mention_analyzer: "LLMTextCompletionProgram",
            pair_analyzer: "LLMTextCompletionProgram",
            verbose: bool=False,
            **kwargs: Any,
        ) -> None:
        super().__init__(*args, **kwargs)
        self.mention_validator = mention_validator
        self.mention_analyzer  = mention_analyzer
        self.pair_analyzer     = pair_analyzer
        self.verbose = verbose

    @step
    async def validate_inputs(self, ctx: Context, ev: StartEvent) -> MentionAnalysisEvent | FinalEvent:
        if self.verbose: print('validating inputs')
        mention_a, mention_b = ev.input
        await ctx.set('mentions', [mention_a, mention_b])
        output = self.mention_validator(mention=mention_a)
        if output.valid=='no':
            await ctx.set('no_label_cause', f"mention a ('{mention_a}') not a valid social group mention.")
            return FinalEvent()
        output = self.mention_validator(mention=mention_b)
        if output.valid=='no':
            await ctx.set('no_label_cause', f"mention b ('{mention_b}') not a valid social group mention.")
            return FinalEvent()
        return MentionAnalysisEvent()

    @step
    async def analyze_inputs(self, ctx: Context, ev: MentionAnalysisEvent) -> PairAnalysisEvent | FinalEvent:
        if self.verbose: print('analyzing inputs')
        (mention_a, mention_b) = await ctx.get('mentions')
        output = self.mention_analyzer(mention=mention_a)
        await ctx.set('analysis_a', output.analysis)
        output = self.mention_analyzer(mention=mention_b)
        await ctx.set('analysis_b', output.analysis)
        return PairAnalysisEvent()
        
    @step
    async def analyze_pair(self, ctx: Context, ev: PairAnalysisEvent) -> FinalEvent:
        if self.verbose: print('analyzing pair')
        (mention_a, mention_b) = await ctx.get('mentions')
        analysis_a = await ctx.get('analysis_a')
        analysis_b = await ctx.get('analysis_b')
        output = self.pair_analyzer(
            mention_a=mention_a, analysis_a=analysis_a, 
            mention_b=mention_b, analysis_b=analysis_b
        )
        await ctx.set('pair_reasoning', output.reasoning)
        await ctx.set('pair_classification', output.decision)
        return FinalEvent()
    
    @step
    async def return_results(self, ctx: Context, ev: FinalEvent) -> StopEvent:
        if self.verbose: print('returning results')
        mentions = await ctx.get('mentions')
        analyses = [
            await ctx.get('analysis_a', default=None),
            await ctx.get('analysis_b', default=None)
        ]
        pair_reasoning = await ctx.get('pair_reasoning', default=None)
        label = await ctx.get('pair_classification', default=None)
        no_label_cause = await ctx.get('no_label_cause', default=None)
        output = WorkflowOutput(
            inputs=mentions,
            analyses=analyses,
            reasoning=pair_reasoning,
            label=label,
            no_label_cause=no_label_cause
        )
        return StopEvent(result=output)

# ## Inference

# ### load data

import pandas as pd

df = pd.read_csv(args.input_file, sep="\t")

if args.test:
    df = df.head(min(args.n_test, len(df)))

# ### Init the workflow

w = MentionPairClassificationWorkflow(
    mention_validator=validator, 
    mention_analyzer=analyzer, 
    pair_analyzer=pair_analyzer, 
    timeout=60.0,
    verbose=False
)

# ### Run

async def process(row):
    out = row.to_dict()
    try:
        response = await w.run(input=[row.mention_a, row.mention_b])
    except Exception as e:
        print('WARNING: could not process mention pair. Reason:', str(e))
        out['error'] = str(e)
    else:
        out.update(response.model_dump())
    return out

outputs = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    output = asyncio.run(process(row))
    outputs.append(output)

# ## Write to disk
if args.output_file is None:
    print(outputs)
    sys.exit(0)

if os.path.exists(args.output_file) and not args.overwrite_output_file:
    raise FileExistsError(f'file "{args.output_file}" already exists. Set --overwrite_output_file to overwrite')
    sys.exit(1)

dest = os.path.dirname(args.output_file)
os.makedirs(dest, exist_ok=True)

try:
    write_jsonlines(outputs, file=args.output_file, overwrite=args.overwrite_output_file)
except Exception as e:
    raise e
    sys.exit(1)
else:
    sys.exit(0)


