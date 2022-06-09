
from typing import List, Dict, Tuple, Generator

import pandas as pd
from tqdm import tqdm

from qanom.annotations.common import read_annot_csv, read_csv, get_sent_map, get_predicate_idx_label
from qanom.annotations.decode_encode_answers import NO_RANGE, Argument, Role, Question, Response, decode_response
from qanom.evaluation.alignment import find_matches, align_by_argument
from qanom.evaluation.metrics import Metrics, BinaryClassificationMetrics


def to_qa_pair(roles: List[Role]) -> List[Tuple[Argument, Question]]:
    return [(arg, role.question) for role in roles for arg in role.arguments]


def build_all_qa_pairs(sys_roles: List[Role],
                       grt_roles: List[Role],
                       sys_to_grt_matches: Dict[Argument, Argument]):
    grt_qa_pairs = to_qa_pair(grt_roles)
    sys_qa_pairs = to_qa_pair(sys_roles)

    grt_qa_pairs = pd.DataFrame(grt_qa_pairs, columns=["grt_arg", "grt_role"])
    sys_qa_pairs = pd.DataFrame(sys_qa_pairs, columns=["sys_arg", "sys_role"])
    # Dictionary mapping with None values
    sys_qa_pairs['grt_arg'] = sys_qa_pairs.sys_arg.apply(sys_to_grt_matches.get)
    all_qa_pairs = pd.merge(sys_qa_pairs, grt_qa_pairs, on="grt_arg", how="outer")
    all_qa_pairs.grt_arg.fillna(NO_RANGE, inplace=True)
    all_qa_pairs.sys_arg.fillna(NO_RANGE, inplace=True)

    return all_qa_pairs


def filter_ids(df, row):
    idx_lbl = get_predicate_idx_label(df)
    return (df.qasrl_id == row.qasrl_id) & (df[idx_lbl] == row[idx_lbl])


def fill_answer(arg: Argument, tokens: List[str]):
    if arg == NO_RANGE:
        return NO_RANGE
    return " ".join(tokens[arg[0]: arg[1]])


def eval_datasets(sys_df, grt_df, sent_map= None) \
        -> Tuple[Metrics, Metrics, Metrics, BinaryClassificationMetrics, pd.DataFrame]:
    if not sent_map:
        annot_df = pd.concat([sys_df[['qasrl_id', 'sentence']], grt_df[['qasrl_id', 'sentence']]])
        sent_map = get_sent_map(annot_df)
    arg_metrics = Metrics.empty()
    labeled_arg_metrics = Metrics.empty()
    role_metrics = Metrics.empty()
    is_nom_counts = BinaryClassificationMetrics.empty()
    all_matchings = []
    for key, sys_response, grt_response in tqdm(yield_paired_predicates(sys_df, grt_df), leave=False):
        qasrl_id, target_idx = key
        tokens = sent_map[qasrl_id]
        local_arg_metric, local_labeled_arg_metric, local_role_metric, local_is_nom_metric, sys_to_grt = \
            evaluate_response(sys_response, grt_response)
        arg_metrics += local_arg_metric
        labeled_arg_metrics += local_labeled_arg_metric
        role_metrics += local_role_metric
        is_nom_counts += local_is_nom_metric
        all_args = build_all_qa_pairs(sys_response.roles, grt_response.roles, sys_to_grt)
        all_args['qasrl_id'] = qasrl_id
        all_args['target_idx'] = target_idx
        all_args['grt_arg_text'] = all_args.grt_arg.apply(fill_answer, tokens=tokens)
        all_args['sys_arg_text'] = all_args.sys_arg.apply(fill_answer, tokens=tokens)
        all_matchings.append(all_args)

    # when all_matching is empty, return empty DataFrame
    if not all_matchings:
        all_matchings = pd.DataFrame()
    else:
        all_matchings = pd.concat(all_matchings)
        all_matchings = all_matchings[['grt_arg_text', 'sys_arg_text',
                                       'grt_role', 'sys_role',
                                       'grt_arg', 'sys_arg',
                                       'qasrl_id', 'target_idx']]

    return arg_metrics, labeled_arg_metrics, role_metrics, is_nom_counts, all_matchings


def get_recall_and_precision_mistakes(sys_df, grt_df) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Return dfs of recall and precision mistakes. """
    all_matchings: pd.DataFrame = eval_datasets(sys_df, grt_df)[-1]
    # take all recall errors - roles which are NA in sys_roles in the alignment
    recall_mistakes = all_matchings[all_matchings['sys_role'].isna()].copy()
    recall_mistakes['question'] = recall_mistakes['grt_role'].apply(lambda role: role.text)
    # take all precision errors - roles which are NA in grt_roles in the alignment
    precision_mistakes = all_matchings[all_matchings['grt_role'].isna()].copy()
    precision_mistakes['question'] = precision_mistakes['sys_role'].apply(lambda role: role.text)
    # now produce the subsets from original DataFrames
    cols = ['qasrl_id', 'target_idx', 'question']
    recall_mistakes_df: pd.DataFrame = grt_df.merge(recall_mistakes[cols], on=cols, how="inner").drop_duplicates(cols)
    precision_mistakes_df: pd.DataFrame = sys_df.merge(precision_mistakes[cols], on=cols, how="inner").drop_duplicates(cols)

    return recall_mistakes_df, precision_mistakes_df

def eval_datasets_arg_f1(sys_df, grt_df) -> float:
    arg_metrics = eval_datasets(sys_df, grt_df)[0]
    return arg_metrics.f1()


def yield_paired_predicates(sys_df: pd.DataFrame, grt_df: pd.DataFrame) -> Generator[Tuple[Tuple[str,int],Response,Response], None, None]:
    grt_predicate_ids = grt_df[['qasrl_id', 'target_idx']].drop_duplicates()
    sys_predicate_ids = sys_df[['qasrl_id', 'target_idx']].drop_duplicates()
    # Include only those predicates which are both in grt and in sys
    predicate_ids = pd.merge(grt_predicate_ids, sys_predicate_ids, how='inner')
    for idx, row in predicate_ids.iterrows():
        sys_qa_pairs = sys_df[filter_ids(sys_df, row)].copy()
        grt_qa_pairs = grt_df[filter_ids(grt_df, row)].copy()
        sys_response = decode_response(sys_qa_pairs)
        grt_response = decode_response(grt_qa_pairs)
        yield (row.qasrl_id, row.target_idx), sys_response, grt_response


def evaluate_response(sys_response: Response,
                      grt_response: Response):
    sys_roles: List[Role] = sys_response.roles
    grt_roles: List[Role] = grt_response.roles
    sys_to_grt = find_matches(sys_response.all_args(), grt_response.all_args())

    is_nom_metrics = BinaryClassificationMetrics.simple_boolean_decision(sys_response.is_verbal, grt_response.is_verbal)

    # todo decide on evaluation of roles where is_verbal mismatch - should the roles be included in the role_count metric?
    # Currently excluding these mismatches from the arg & roles metrics
    if is_nom_metrics.errors() == 0:
        arg_metrics = eval_arguments(grt_roles, sys_roles, sys_to_grt)
        labeled_arg_metrics = eval_labeled_arguments(grt_roles, sys_roles, sys_to_grt)
        role_metrics = eval_roles(grt_roles, sys_roles, sys_to_grt)
    else:
        arg_metrics = Metrics.empty()
        labeled_arg_metrics = Metrics.empty()
        role_metrics = Metrics.empty()

    return arg_metrics, labeled_arg_metrics, role_metrics, is_nom_metrics, sys_to_grt


def count_arguments(roles: List[Role]):
    return sum(len(role.arguments) for role in roles)


def eval_arguments(grt_roles: List[Role], sys_roles: List[Role], sys_to_grt: Dict[Argument, Argument]) -> Metrics:
    tp_arg_count = len(sys_to_grt)
    fp_arg_count = count_arguments(sys_roles) - tp_arg_count
    fn_arg_count = count_arguments(grt_roles) - tp_arg_count
    return Metrics(tp_arg_count, fp_arg_count, fn_arg_count)


def eval_labeled_arguments(grt_roles: List[Role], sys_roles: List[Role], sys_to_grt: Dict[Argument, Argument]) -> Metrics:
    """ LA metric - Labeled Argument match - spans overlap and questions are equivalent. """
    tp_arg_count = count_labeled_arg_matches(grt_roles, sys_roles, sys_to_grt)
    fp_arg_count = count_arguments(sys_roles) - tp_arg_count
    fn_arg_count = count_arguments(grt_roles) - tp_arg_count
    return Metrics(tp_arg_count, fp_arg_count, fn_arg_count)


def count_labeled_arg_matches(grt_roles: List[Role], sys_roles: List[Role], sys_to_grt: Dict[Argument, Argument]) -> int:
    """ Count the number of labeled_matching among argument-match (TP) -
    i.e. only if corresponding questions are equivalent. """
    # for each argument in sys_to_grt, find corresponding question from its role
    def find_role(arg: Argument, roles: List[Role]) -> Role:
        for r in roles:
            if arg in r.arguments:
                return r

    sys_arg2q: Dict[Argument, Role] = {arg : find_role(arg, sys_roles)
                                       for arg in sys_to_grt.keys()}
    grt_arg2q: Dict[Argument, Role] = {arg : find_role(arg, grt_roles)
                                       for arg in sys_to_grt.values()}
    from qanom.evaluation.roles import is_equivalent_question

    def is_labeled_arg_match(sys_arg: Argument, grt_arg: Argument) -> bool:
        sys_question = sys_arg2q[sys_arg].question
        grt_question = grt_arg2q[grt_arg].question
        return is_equivalent_question(sys_question, grt_question)

    return sum(1
               for arg_match in sys_to_grt.items()
               if is_labeled_arg_match(*arg_match))


def eval_roles(grt_roles: List[Role],
               sys_roles: List[Role],
               sys_to_grt: Dict[Argument, Argument]) -> Metrics:
    alignemnt = align_by_argument(grt_roles, sys_roles, sys_to_grt)
    tp, fp, fn = 0, 0, 0
    for grt_role in grt_roles:
        if alignemnt.has_single_alignment(grt_role, is_grt=True):
            tp += 1
        else:
            fn += 1
    for sys_role in sys_roles:
        if not alignemnt.has_single_alignment(sys_role, is_grt=False):
            fp += 1
    return Metrics(tp, fp, fn)


def print_system_evaluation(system_df: pd.DataFrame, ground_truth_df: pd.DataFrame):
    arg, larg, role, isnom, _ = eval_datasets(system_df, ground_truth_df)

    from qanom.annotations.analyze import print_annot_statistics
    print("System prediction statistics:")
    print_annot_statistics(system_df)
    print("**************************")

    print("Ground Truth statistics:")
    print_annot_statistics(system_df)
    print("**************************")

    print("\n\t\t\t\tPrecision\tRecall\tF1")
    print(f"arg-f1 \t\t\t {arg.prec():.2f}\t{arg.recall():.2f}\t{arg.f1():.4f}")
    print(f"labeled-arg-f1 \t {larg.prec():.2f}\t{larg.recall():.2f}\t{larg.f1():.4f}")
    print(f"role-f1 \t\t {role.prec():.2f}\t{role.recall():.2f}\t{role.f1():.4f}")
    print(f"is-verbal \t\t {isnom.prec():.2f}\t{isnom.recall():.2f}\t{isnom.f1():.4f}")
    print(f"is-verbal (accuracy): \t {isnom.accuracy():.4f}    for {isnom.instances()} pairwise comparisons.")



def main(proposed_path: str, reference_path: str, sentences_path: str):
    if sentences_path:
        sent_df = read_csv(sentences_path)
        sent_map = dict(zip(sent_df.qasrl_id, sent_df.tokens.apply(str.split)))
    else:
        sent_map = None

    sys_df = read_annot_csv(proposed_path)
    grt_df = read_annot_csv(reference_path)
    print_system_evaluation(sys_df, grt_df)

