from argparse import ArgumentParser
from dataclasses import astuple
from typing import List, Dict, Generator

import numpy as np
import pandas as pd
from tqdm import tqdm

from annotations_evaluations.common import Question, Role, QUESTION_FIELDS, Argument
from annotations_evaluations.decode_encode_answers import NO_RANGE, decode_qasrl
from annotations_evaluations.evaluate import evaluate, Metrics


def to_arg_roles(roles: List[Role]):
    return [(arg, role.question) for role in roles for arg in role.arguments]


def build_all_arg_roles(sys_roles: List[Role],
                        grt_roles: List[Role],
                        sys_to_grt_matches: Dict[Argument, Argument]):
    grt_arg_roles = to_arg_roles(grt_roles)
    sys_arg_roles = to_arg_roles(sys_roles)

    grt_arg_roles = pd.DataFrame(grt_arg_roles, columns=["grt_arg", "grt_role"])
    sys_arg_roles = pd.DataFrame(sys_arg_roles, columns=["sys_arg", "sys_role"])
    # Dictionary mapping with None values
    sys_arg_roles['grt_arg'] = sys_arg_roles.sys_arg.apply(sys_to_grt_matches.get)
    all_arg_roles = pd.merge(sys_arg_roles, grt_arg_roles, on="grt_arg", how="outer")
    all_arg_roles.grt_arg.fillna(NO_RANGE, inplace=True)
    all_arg_roles.sys_arg.fillna(NO_RANGE, inplace=True)

    return all_arg_roles


def filter_ids(df, row):
    return (df.qasrl_id == row.qasrl_id) & (df.verb_idx == row.verb_idx)


def fill_answer(arg: Argument, tokens: List[str]):
    if arg == NO_RANGE:
        return NO_RANGE
    return " ".join(tokens[arg[0]: arg[1]])


def eval_datasets(sys_df, grt_df, sent_map, allow_overlaps: bool):
    arg_counts = np.zeros(3, dtype=np.float32)
    role_counts = np.zeros(3, dtype=np.float32)
    all_matchings = []
    for key, sys_roles, grt_roles in tqdm(yield_paired_predicates(sys_df, grt_df), leave=False):
        qasrl_id, verb_idx = key
        tokens = sent_map[qasrl_id]
        local_arg_counts, local_role_counts, sys_to_grt = evaluate(sys_roles, grt_roles, allow_overlaps)
        arg_counts += np.array(astuple(local_arg_counts))
        role_counts += np.array(astuple(local_role_counts))
        all_args = build_all_arg_roles(sys_roles, grt_roles, sys_to_grt)
        all_args['qasrl_id'] = qasrl_id
        all_args['verb_idx'] = verb_idx
        all_args['grt_arg_text'] = all_args.grt_arg.apply(fill_answer, tokens=tokens)
        all_args['sys_arg_text'] = all_args.sys_arg.apply(fill_answer, tokens=tokens)
        all_matchings.append(all_args)

    all_matchings = pd.concat(all_matchings)
    all_matchings = all_matchings[['grt_arg_text', 'sys_arg_text',
                                   'grt_role', 'sys_role',
                                   'grt_arg', 'sys_arg',
                                   'qasrl_id', 'verb_idx']]
    arg_metrics = Metrics(*arg_counts)
    role_metrics = Metrics(*role_counts)

    return arg_metrics, role_metrics, all_matchings


def main(sentences_path: str, proposed_path: str, reference_path: str):
    sent_df = pd.read_csv(sentences_path)
    sent_map = dict(zip(sent_df.qasrl_id, sent_df.tokens.apply(str.split)))

    sys_df = decode_qasrl(pd.read_csv(proposed_path))
    grt_df = decode_qasrl(pd.read_csv(reference_path))
    arg, role, _ = eval_datasets(sys_df, grt_df, sent_map, allow_overlaps=False)

    print("ARGUMENT: Prec/Recall ", arg.prec(), arg.recall(), arg.f1())
    print("ROLE: Prec/Recall ", role.prec(), role.recall(), role.f1())


def yield_paired_predicates(sys_df: pd.DataFrame, grt_df: pd.DataFrame):
    predicate_ids = grt_df[['qasrl_id', 'verb_idx']].drop_duplicates()
    for idx, row in predicate_ids.iterrows():
        sys_arg_roles = sys_df[filter_ids(sys_df, row)].copy()
        grt_arg_roles = grt_df[filter_ids(grt_df, row)].copy()
        sys_roles = list(yield_roles(sys_arg_roles))
        grt_roles = list(yield_roles(grt_arg_roles))
        yield (row.qasrl_id, row.verb_idx), sys_roles, grt_roles


def question_from_row(row: pd.Series) -> Question:
    question_as_dict = {question_field: row[question_field]
                        for question_field in QUESTION_FIELDS}
    question_as_dict['text'] = row.question
    return Question(**question_as_dict)


def yield_roles(predicate_df: pd.DataFrame) -> Generator[Role, None, None]:
    for row_idx, role_row in predicate_df.iterrows():
        question = question_from_row(role_row)
        arguments: List[Argument] = role_row.answer_range
        yield Role(question, tuple(arguments))


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("sentences_path")
    ap.add_argument("sys_path")
    ap.add_argument("ground_truth_path")
    args = ap.parse_args()
    main(args.sentences_path, args.sys_path, args.ground_truth_path)