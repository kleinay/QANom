from dataclasses import dataclass
from typing import Tuple

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


@dataclass(frozen=True)
class Role:
    question: Question
    arguments: Tuple[Argument, ...]

    def text(self):
        return self.question.text



