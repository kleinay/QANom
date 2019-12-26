from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd


def normalize(lst):
    a = np.array(lst)
    return a / sum(a)


def read_dir_of_csv(dir_path: str) -> pd.DataFrame:
    """ Concatenate all csv files in directory into one DataFrame """
    import os
    dfs, sections = zip(*[(read_csv(os.path.join(dir_path, fn)), fn.rstrip(".csv"))
                          for fn in os.listdir(dir_path) if fn.endswith(".csv")])
    return pd.concat(dfs, ignore_index=True, keys=sections, sort=True)


def read_csv(file_path: str) -> pd.DataFrame:
    from annotations_evaluations.decode_encode_answers import decode_qasrl
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="Latin-1")
    return decode_qasrl(df)

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



