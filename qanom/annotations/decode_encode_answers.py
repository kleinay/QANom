"""
For clarification:  we use the terms "encoding" and "decoding" for the bidirectional path between two formats
of the qanom data. This script and the functions within it refer to the *exported CSV file* as the "encoded" format,
and to the code-internal annot_df format (in which some columns values are objects\instances, and not strings)
as the "decoded" format. The decoded format is easily transferrable between DataFrame and the QANom class hierarchy
defined in this file (Response, Role and Question. Argument is actually just an alias for Tuple[int, int]).
Main differences:
* a single answer span:
 ** encoded: answer-spans are strings in the form f"{start}:{end}" , e.g. "2:3"
 ** decoded: answer-spans are objects of type Tuple[int, int] , e.g. (2,3)
* answer / answer_range columns:
 ** encoded: "he~!~John", "2:3~!~7:8"
 ** decoded: ["he", "John"], [(2,3), (7,8)]
"""

from dataclasses import dataclass, field, asdict
from typing import List, Iterable, Tuple, Generator

import pandas as pd

SPAN_SEPARATOR = "~!~"
NO_RANGE = ""
INVALID = "INVALID"

Argument = Tuple[int, int]
def arg_length(argument: Argument) -> int:
    return argument[1]-argument[0]

QUESTION_FIELDS = ['wh', 'subj', 'obj', 'aux', 'verb_prefix', 'verb_slot_inflection', 'prep', 'obj2', 'is_passive', 'is_negated']


@dataclass(frozen=True)
class Question:
    text: str
    wh: str
    subj: str
    obj: str
    aux: str
    verb_prefix: str
    verb_slot_inflection: str
    prep: str
    obj2: str
    is_passive: bool
    is_negated: bool

    def __str__(self):
        return self.text

    def isEmpty(self) -> bool:
        return self.text == ""

    @classmethod
    def empty(cls):
        return Question("", "", "", "", "", "", "", "", "", False, False)

    @classmethod
    def text_only(cls, question_text):
        return Question(question_text, "", "", "", "", "", "", "", "", False, False)


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
    roles: List[Role] = field(default_factory=list)

    def all_args(self) -> List[Argument]:
        return [arg for role in self.roles for arg in role.arguments]


def is_invalid_range(r: Argument):
    return r[0] == NO_RANGE


def encode_argument(arg: Iterable[Argument]) -> str:
    if not arg or list(arg)[0] == NO_RANGE:
        return NO_RANGE

    return SPAN_SEPARATOR.join([encode_span(span) for span in arg])


def encode_span(span: Argument):
    if span == NO_RANGE or not span:
        return NO_RANGE
    return "{}:{}".format(span[0], span[1])


def encode_argument_text(arg_text: Iterable[str]):
    return SPAN_SEPARATOR.join(arg_text)


def encode_argument_text_from_spans(arguments: Iterable[Argument], tokens: List[str]) -> str:
    """ Return a string for the CSV field 'answer' """
    return SPAN_SEPARATOR.join([span_to_text(span, tokens) for span in arguments])


def span_to_text(span: Argument, tokens: List[str]) -> str:
    if type(span) is not tuple:
        return ""
    span_start, span_end = span
    return " ".join(tokens[span_start: span_end])


def decode_span(span_str: str) -> Argument:
    splits = span_str.split(":")

    return int(splits[0]), int(splits[1])


def decode_argument(arg_str: str) -> List[Argument]:
    if not arg_str:
        return []
    ranges = arg_str.split(SPAN_SEPARATOR)
    if ranges[0] == NO_RANGE:
        return ranges

    ranges = [decode_span(span_str) for span_str in ranges]
    return ranges


def decode_qasrl(qasrl_df: pd.DataFrame) -> pd.DataFrame:
    # WHY WHY WHY WE HAVE NULLS??? (see below why)
    # qasrl_df.dropna(subset=['qasrl_id', 'target_idx', 'question'], inplace=True)
    cols = set(qasrl_df.columns)
    answer_range_cols = set([col for col in cols if "answer_range" in col])
    answer_cols = set([col for col in cols if "answer" in col]) - answer_range_cols
    # We sometimes have null values in answer (but not in answer range) if the answer was actually the string:
    # NA or anything in the default NA values for read_csv
    ## common NA values
    # no longer excluding inf representations
    # '1.#INF','-1.#INF', '1.#INF000000',
    # _NA_VALUES = set([
    # '-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A',
    # 'N/A', 'n/a', 'NA', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', ''
    # ])
    # drop them, for now

    #if answer_cols:
    #    qasrl_df.dropna(subset=answer_cols, inplace=True)

    for c in answer_cols:
        # qasrl_df[c] = qasrl_df[c].astype(str)
        qasrl_df[c].fillna("", inplace=True)
        qasrl_df[c] = qasrl_df[c].apply(lambda a: a.split(SPAN_SEPARATOR))
    for c in answer_range_cols:
        # qasrl_df[c] = qasrl_df[c].astype(str)
        qasrl_df[c].fillna("", inplace=True)
        qasrl_df[c] = qasrl_df[c].apply(decode_argument)

    for c in QUESTION_FIELDS:
        if c in qasrl_df:
            qasrl_df[c].fillna("", inplace=True)

    from qanom.annotations.common import set_key_column
    set_key_column(qasrl_df)
    return qasrl_df


def encode_qasrl(qasrl_df) -> pd.DataFrame:
    for_csv = qasrl_df.copy()
    cols = set(qasrl_df.columns)
    answer_range_cols = set([col for col in cols if "answer_range" in col])
    answer_cols = set([col for col in cols  if "answer" in col]) - answer_range_cols

    for c in answer_range_cols:
        for_csv[c] = for_csv[c].apply(encode_argument)
    for c in answer_cols:
        for_csv[c] = for_csv[c].apply(encode_argument_text)
    return for_csv


def encode_response(response: Response, sentence_str: str, **kwargs) -> pd.DataFrame:
    """
    should return a dataFrame capturing the annotations in Response.
    of course, this df will only include annotation-columns and not (other) data- or metadata- columns.
    kwargs can be used to insert other information to the columns of the output dataframe - these key:value
    pairs will be part of the information that is the same in all the rows of this response (e.g. target_idx).
    """
    common_for_rows = dict({'is_verbal' : response.is_verbal, 'verb_form': response.verb_form}, **kwargs)
    def role2rowDict(role: Role) -> dict:
        questionDict = asdict(role.question)
        questionDict['question'] = questionDict.pop('text')
        qaDict = dict(questionDict,
                      answer_range=encode_argument(role.arguments),
                      answer=encode_argument_text_from_spans(role.arguments, sentence_str.split(" ")))
        rowDict = dict(qaDict, **common_for_rows)
        return rowDict

    rolesToEncode = response.roles if response.roles else [Role(Question.empty(), tuple())]
    lstOfRowDicts = list(map(role2rowDict, rolesToEncode))
    return pd.DataFrame(lstOfRowDicts)


def decode_response(predicate_df: pd.DataFrame) -> Response:
    roles = list(yield_roles(predicate_df))
    is_verbal = predicate_df['is_verbal'].iloc[0] if len(predicate_df)>0 else True
    verb_form = predicate_df['verb_form'].iloc[0] if len(predicate_df)>0 else ""
    return Response(is_verbal,
                    verb_form,
                    roles)


def question_from_row(row: pd.Series) -> Question:
    question_as_dict = {question_field: row[question_field]
                        for question_field in QUESTION_FIELDS}
    question_as_dict['text'] = row.question if not pd.isnull(row.question) else ""
    return Question(**question_as_dict)


def yield_roles(predicate_df: pd.DataFrame) -> Generator[Role, None, None]:
    for row_idx, role_row in predicate_df.iterrows():
        if any(q_field not in predicate_df.columns for q_field in QUESTION_FIELDS):
            # this is not a qanom annotation file, but rather other format (e.g. NomBank csv file).
            # Thus, take only question text as Question
            question = Question.text_only(role_row.question)
        else:
            question = question_from_row(role_row)
        arguments: List[Argument] = role_row.answer_range
        # for No-QA-Applicable, yield no role (empty list of roles)
        if not question.isEmpty():
            yield Role(question, tuple(arguments))