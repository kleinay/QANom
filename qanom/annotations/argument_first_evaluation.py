from argparse import ArgumentParser
from typing import List, Dict, Tuple, Generator

import pandas as pd
from tqdm import tqdm

from annotations.common import read_annot_csv, read_csv
from annotations.decode_encode_answers import NO_RANGE, decode_qasrl, Argument, Role, Response, \
    decode_response
from annotations.evaluate import evaluate, Metrics, BinaryClassificationMetrics


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


def eval_datasets(sys_df, grt_df, sent_map, allow_overlaps: bool) \
        -> Tuple[Metrics, Metrics, BinaryClassificationMetrics, pd.DataFrame]:
    if not sent_map:
        annot_df = pd.merge(sys_df[['qasrl_id', 'sentence']], grt_df[['qasrl_id', 'sentence']])
        import annotations.evaluate_inter_annotator
        sent_map = annotations.evaluate_inter_annotator.get_sent_map(annot_df)
    arg_metrics = Metrics.empty()
    role_metrics = Metrics.empty()
    is_nom_counts = BinaryClassificationMetrics.empty()
    all_matchings = []
    for key, sys_response, grt_response in tqdm(yield_paired_predicates(sys_df, grt_df), leave=False):
        qasrl_id, verb_idx = key
        tokens = sent_map[qasrl_id]
        local_arg_metric, local_role_metric, local_is_nom_metric, sys_to_grt = evaluate(sys_response, grt_response, allow_overlaps)
        arg_metrics += local_arg_metric
        role_metrics += local_role_metric
        is_nom_counts += local_is_nom_metric
        all_args = build_all_arg_roles(sys_response.roles, grt_response.roles, sys_to_grt)
        all_args['qasrl_id'] = qasrl_id
        all_args['verb_idx'] = verb_idx
        all_args['grt_arg_text'] = all_args.grt_arg.apply(fill_answer, tokens=tokens)
        all_args['sys_arg_text'] = all_args.sys_arg.apply(fill_answer, tokens=tokens)
        all_matchings.append(all_args)

    # todo verify fix bug - when all_matching is empty, return empty DataFrame
    if not all_matchings:
        all_matchings = pd.DataFrame()
    else:
        all_matchings = pd.concat(all_matchings)
        all_matchings = all_matchings[['grt_arg_text', 'sys_arg_text',
                                       'grt_role', 'sys_role',
                                       'grt_arg', 'sys_arg',
                                       'qasrl_id', 'verb_idx']]

    return arg_metrics, role_metrics, is_nom_counts, all_matchings


def main(sentences_path: str, proposed_path: str, reference_path: str):
    sent_df = read_csv(sentences_path)
    sent_map = dict(zip(sent_df.qasrl_id, sent_df.tokens.apply(str.split)))

    sys_df = read_annot_csv(proposed_path)
    grt_df = read_annot_csv(reference_path)
    sys_df = decode_qasrl(sys_df)
    grt_df = decode_qasrl(grt_df)
    arg, role, isnom, _ = eval_datasets(sys_df, grt_df, sent_map, allow_overlaps=False)

    print("ARGUMENT: Prec/Recall ", arg.prec(), arg.recall(), arg.f1())
    print("ROLE: Prec/Recall ", role.prec(), role.recall(), role.f1())
    print("NOM-IDENT: Prec/Recall ", isnom.prec(), isnom.recall(), isnom.f1())
    return (arg, role, isnom)

def yield_paired_predicates(sys_df: pd.DataFrame, grt_df: pd.DataFrame) -> Generator[Tuple[Tuple[str,int],Response,Response], None, None]:
    grt_predicate_ids = grt_df[['qasrl_id', 'verb_idx']].drop_duplicates()
    sys_predicate_ids = sys_df[['qasrl_id', 'verb_idx']].drop_duplicates()
    # Include only those predicates which are both in grt and in sys
    predicate_ids = pd.merge(grt_predicate_ids, sys_predicate_ids, how='inner')
    for idx, row in predicate_ids.iterrows():
        sys_arg_rows = sys_df[filter_ids(sys_df, row)].copy()
        grt_arg_rows = grt_df[filter_ids(grt_df, row)].copy()
        sys_response = decode_response(sys_arg_rows)
        grt_response = decode_response(grt_arg_rows)
        yield (row.qasrl_id, row.verb_idx), sys_response, grt_response


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("sentences_path")
    ap.add_argument("sys_path")
    ap.add_argument("ground_truth_path")
    args = ap.parse_args()
    main(args.sentences_path, args.sys_path, args.ground_truth_path)