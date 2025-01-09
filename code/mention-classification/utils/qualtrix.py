from typing import Union, List, Dict, Optional

def only_text(question_text, question_id: Optional[str]=None) -> str:
    string = '[[Question:Text]]\n'
    if question_id:
        string += f'[[ID:{question_id}]]\n'
    string += question_text + '\n'
    return string

def mc_question(
    question_text,
    choices: Union[List[str], Dict[int, str]],
    question_id: Optional[str]=None,
    multiple_choice: bool=False,
    horizontal: bool=False
) -> str:
    string = '[[Question:MC:SingleAnswer]]' if not multiple_choice else '[[Question:MC:MultipleAnswer]]'
    if horizontal:
        string = string.replace(']]', ':Horizontal]]')
    string += '\n'
    if question_id:
        string += f'[[ID:{question_id}]]\n'
    string += question_text + '\n'
    string += '[[Choices]]' if isinstance(choices, list) else '[[AdvancedChoices]]'
    string += '\n'
    choices = choices if isinstance(choices, list) else [f'[[Choice:{k}]]\n{v}' for k, v in choices.items()]
    choices = '\n'.join(choices)
    string += choices + '\n'
    return string

def matrix_question(
    question_text,
    choices: Union[List[str], Dict[int, str]],
    answers: Union[List[str], Dict[int, str]],
    question_id: Optional[str]=None,
    multiple_choice: bool=False,
    horizontal: bool=False
) -> str:
    string = '[[Question:Matrix:MultipleAnswer]]' if multiple_choice else '[[Question:Matrix:SingleAnswer]]'
    if horizontal:
        string = string.replace(']]', ':Horizontal]]')
    string += '\n'
    if question_id:
        string += f'[[ID:{question_id}]]\n'
    string += question_text + '\n'
    string += '[[Choices]]' if isinstance(choices, list) else '[[AdvancedChoices]]'
    string += '\n'
    choices = choices if isinstance(choices, list) else [f'[[Choice:{k}]]\n{v}' for k, v in choices.items()]
    choices = '\n'.join(choices)
    string += choices + '\n'
    string += '[[Answers]]' if isinstance(answers, list) else '[[AdvancedAnswers]]'
    string += '\n'
    answers = answers if isinstance(answers, list) else [f'[[Answer:{k}]]\n{v}' for k, v in answers.items()]
    answers = '\n'.join(answers)
    string += answers + '\n'
    return string

def text_entry(question_text, question_id: Optional[str]=None) -> str:
    string = '[[Question:TextEntry:SingleLine]]\n'
    if question_id:
        string += f'[[ID:{question_id}]]\n'
    string += question_text + '\n'
    return string

