PERSONA = """\
You are an objective and detail-oriented social scientist, skilled in analyzing language and identifying discursive constructions within texts.\
"""

MAIN_TASK_PROMPT = """\
Your task is to **identify mentions of social groups** in a text, including direct mentions using groups' names, descriptions of groups based on their members' shared characteristics, and implicit references that imply group boundaries.\
"""

SOCIAL_GROUP_DEFINITION = """\
A **social group** is a collection of individuals who share common characteristics, traits, attitudes, habits, experiences, etc. and can be explicitly named, described, or implied in text by referring to its shared traits.\
"""

SOCIAL_GROUP_MENTION_DEFINITION = """\
A **social group mention** is a phrase that refers to or discursively constructs a social group by explicitly naming it, describing its members' shared characteristics, or implying its existence through contextual cues.\
"""

ATTRIBUTE_CLASSIFICATION_TASK_PROMPT_TEMPLATE = f"""\
Below, you see the text of a phrase that refers to a social group ("group mention") and the sentence in which it occurs.

Your task is to classify whether the group mention is based on an attribute related to {{attribute_name}}. That is, you must determine whether the phrase describing the group mentioned characterizes the groups members as sharing attributes related to {{attribute_name}}.

## Definitions

- {SOCIAL_GROUP_DEFINITION}
- {SOCIAL_GROUP_MENTION_DEFINITION}
- Groups characterized by attribute(s) related to **{{attribute_name}}** are {{attribute_definition}}.

## Step-by-step task instructions

1. Read the group mention carefully.
2. Think about whether the group mentioned or described in the sentence uses an attribute related to {{attribute_name}} to define the focal group. 
   Specifically, consider whether the phrase describing the group mentioned characterizes the groups members as sharing attributes related to {{attribute_name}}.
   Focus your attention on the specific group mention provided.
   Only choose "yes" if the {{attribute_name}} characteristics are stated _explicitly_ in the group mention.
3. Respond a JSON that records 
    1. a 1-2 sentence summary of your reasoning and
    2. your classification decision ("yes" or "no").

## Input

SENTENCE: \"\"\"{{text}}\"\"\"
GROUP: \"\"\"{{mention}}\"\"\"

## Response format

Return your response as a JSON list dictionary with fields "reasoning" (str) and "classification" (str, "yes" or "no").\
"""
