from dataclasses import dataclass
from typing import Tuple, List

import pandas as pd


def read_csv(file_path: str):
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="Latin-1")

Argument = Tuple[int, int]

QUESTION_FIELDS = ['wh', 'subj', 'obj', 'aux', 'prep', 'obj2', 'is_passive', 'is_negated']

@dataclass(frozen=True)
class Question:
    text: str
    wh: str
    subj: str
    obj: str
    aux: str
    prep: str
    obj2: str
    is_passive: bool
    is_negated: bool

    def __str__(self):
        return self.text

    def isEmpty(self) -> bool:
        return self.text is ""


@dataclass(frozen=True)
class Role:
    question: Question
    arguments: Tuple[Argument, ...]

    def text(self):
        return self.question.text


@dataclass(frozen=True)
class Response:
    is_verbal: bool
    verb_form: str
    roles: List[Role] = None



