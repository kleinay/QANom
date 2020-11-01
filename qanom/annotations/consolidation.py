"""  This script provide the required functionality for consolidating the QANom annotation from multiple generators
 (typically two) into a final, non-redundant annotation. """
from typing import *

import pandas as pd

from qanom import utils
from qanom.annotations import common, decode_encode_answers
from qanom.annotations.decode_encode_answers \
    import Argument, Role, Response, encode_response, arg_length
from qanom.evaluation.metrics import iou

FINAL_COLUMNS = ['qasrl_id', 'sentence', 'target_idx', 'key', 'verb',
                 'worker_id', 'assign_id',
                 'is_verbal', 'verb_form',
                 'question', 'answer_range', 'answer',
                 'wh', 'subj', 'obj', 'obj2', 'aux', 'prep', 'verb_prefix',
                 'is_passive', 'is_negated']
# original annot files include also 'source_assign_id' and 'is_redundant' (for arbitration\validation)

def auto_consolidate_gen_annot_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return consolidated (non-redundant) QANom annotations.
    Algorithm is precision-oriented:
        take conjunction of is-verbal,
        filter only roles (questions) aligned by their answers,
        for answers: take larger set of answers;
            if even - take longer (and select the question corresponding to selected answers).
    :param df: generation annotation DataFrame containing multiple workers' annotation per predicate
    :return: annotation csv (encoded, not decoded) of the consolidated annotations
    """
    from annotations.decode_encode_answers import SPAN_SEPARATOR
    data_columns = ['qasrl_id', 'sentence', 'target_idx', 'key', 'verb', 'verb_form']
    to_be_conjoined_columns = ['worker_id', 'assign_id']
    """ Rest of columns are annotation-columns - they are to be decoded from consolidated Response.
        (Except from 'answer' which requires both answer_range from Response and sentence.) """
    pred_dfs: List[pd.DataFrame] = []
    for key, pred_df in df.groupby('key'):
        responses = {worker: decode_encode_answers.decode_response(worker_df)
                     for worker, worker_df in pred_df.groupby('worker_id')}
        if len(responses)<2:
            # only one generator for predicate
            print(f"Warning: predicate {key} has only one generator. Taking his response as final.")
            consolidated_response = list(responses.values())[0]
        else:
            # two generators = automatically consolidate responses
            consolidated_response = auto_consolidate_predicate(list(responses.values()))
        cons_pred_df = encode_response(consolidated_response, pred_df.sentence.to_list()[0])
        # add other columns to resulting df
        cons_pred_df = cons_pred_df.assign(**{col: pred_df[col].to_list()[0] for col in data_columns})
        cons_pred_df = cons_pred_df.assign(**{col: SPAN_SEPARATOR.join(set(pred_df[col])) for col in to_be_conjoined_columns})
        # re-order columns for convenience
        cons_pred_df = cons_pred_df[FINAL_COLUMNS]
        pred_dfs.append(cons_pred_df)
    out = pd.concat(pred_dfs)
    return out


def get_aligned_roles(resp1: Response, resp2: Response) -> List[Role]:
    # take only aligned arguments along with their full Role (=QA)
    from evaluation.alignment import find_matches
    alignment: Dict[Argument, Argument] = find_matches(resp1.all_args(), resp2.all_args())

    # Helper func:
    def orig_role(arg: Argument, response: Response) -> Role:
        """ get original role (question & argument-set) of arg. """
        roles_containing_arg = list(filter(lambda r: arg in r.arguments, response.roles))
        # shouldn't ever happen due to UI restriction on overlapping arguments
        assert len(
            roles_containing_arg) == 1, f"More or less than one role contains this arg: {arg}, roles: {roles_containing_arg}"
        return roles_containing_arg[0]

    def is_conflicting(role: Role, pre_selected_roles: Set[Role]) -> bool:
        """ Return True if any of role.arguments if conflicting (overlapping) with the pre-selected arguments. """
        overlapping = lambda a1, a2: iou(a1, a2) > 0
        return any(overlapping(arg, pre_arg)
                   for arg in role.arguments
                   for pre_role in pre_selected_roles
                   for pre_arg in pre_role.arguments)

    selected_roles: Set[Role] = set()

    def get_better_role(role1: Role, role2: Role) -> Union[Role, None]:
        """ Choose which role to select for consolidated response (from two aligned roles) """
        # Criteria 0: role doesn't conflict with roles we have previosuly selected
        role1_conflict = is_conflicting(role1, selected_roles)
        role2_conflict = is_conflicting(role2, selected_roles)
        if role1_conflict and role2_conflict:
            return None
        elif role1_conflict:
            return role2
        elif role2_conflict:
            return role1
        # Criteria 1: num of spans in argument-set
        if len(role1.arguments) != len(role2.arguments):
            return role1 if len(role1.arguments) > len(role2.arguments) else role2
        # Criteria 2: (overall) length of arguments
        num_tokens1 = sum(arg_length(arg) for arg in role1.arguments)
        num_tokens2 = sum(arg_length(arg) for arg in role2.arguments)
        if num_tokens1 != num_tokens2:
            return role1 if num_tokens1 > num_tokens2 else role2
        # no other criteria - arbitrarily select
        return role1

    for arg_w1, arg_w2 in alignment.items():
        # choose which role to select for consolidated response
        role1 = orig_role(arg_w1, resp1)
        role2 = orig_role(arg_w2, resp2)
        new_role = get_better_role(role1, role2)
        if new_role:
            selected_roles.add(new_role)
    return list(selected_roles)


def auto_consolidate_predicate(responses: List[Response], method: str = "intersection") -> Response:
    """
    :param responses:
    :param method: "intersection" | "majority"
    :return:
    """
    assert method in ["intersection", "majority"], "'method' must be 'intersection' or 'majority'!"
    if len(responses) == 1:
        return responses[0]
    assert len(responses) > 1, "Got 0 responses to consolidate"

    verb_form = responses[0].verb_form
    is_verbal_decisions = list(map(lambda r: r.is_verbal, responses))
    majority_is_verbal = utils.majority(is_verbal_decisions, whenTie=False)
    has_no_roles = list(map(lambda r: not bool(r.roles), responses))
    # in case no generator produced QAs for this predicate
    if all(has_no_roles):
        return Response(majority_is_verbal, verb_form)

    if len(responses) > 2:
        if method=="intersection":
            # recursive solution - taking only intersection roles
            cons_response1 = auto_consolidate_predicate(responses[:2])
            cons_response2 = auto_consolidate_predicate(responses[2:])
            return auto_consolidate_predicate([cons_response1, cons_response2], method)
        elif method=="majority":
            # todo
            raise NotImplementedError

    # assuming Iterable of 2 responses
    elif len(responses) == 2:
        selected_roles = get_aligned_roles(*responses[:2])
        return Response(majority_is_verbal, verb_form, selected_roles)


def auto_consolidation_iaa_experiment(dup_annot_df: pd.DataFrame) -> float:
    # returns argument F1 agreement
    import evaluation.evaluate_inter_annotator as eia
    desired_workers = ['A1FS8SBR4SDWYG', 'A21LONLNBOB8Q', 'A2A4UAFZ5LW71K', 'AJQGWGESKQT4Y']
    only4w = dup_annot_df[dup_annot_df.worker_id.isin(desired_workers)]
    df = only4w
    df['n_workers'] = df.groupby('key').worker_id.transform(pd.Series.nunique)
    only4w = df[df.n_workers == 4]
    print(f"number of predicates with full annotation of same 4 workers: {common.get_n_predicates(only4w)}")

    grp1, grp2 = ['A1FS8SBR4SDWYG', 'AJQGWGESKQT4Y'], ['A21LONLNBOB8Q', 'A2A4UAFZ5LW71K']
    r2grp = lambda r: "grp1" if r.worker_id in grp1 else "grp2"
    only4w['group'] = only4w.apply(r2grp, axis=1)
    grp2df = dict(list(only4w.groupby('group')))
    # generate consolidated df (automatic) for each group
    cons1_df = auto_consolidate_gen_annot_df(grp2df["grp1"])
    cons2_df = auto_consolidate_gen_annot_df(grp2df["grp2"])
    meta_df = pd.concat([cons1_df, cons2_df], ignore_index=True, sort=False)
    import annotations.decode_encode_answers as decode_encode
    meta_decoded_df = decode_encode.decode_qasrl(meta_df)
    # print IAA
    return eia.evaluate_inter_generator_agreement(meta_decoded_df)
