"""  This script provide the required functionality for consolidating the QANom annotation from multiple generators
 (typically two) into a final, non-redundant annotation. """
from typing import *

import annotations_evaluations.decode_encode_answers
from annotations_evaluations.decode_encode_answers \
    import Argument, Role, Response, encode_response, arg_length
from annotations_evaluations.evaluate import iou
from qanom.annotations_evaluations.common import *

FINAL_COLUMNS = ['qasrl_id', 'sentence', 'verb_idx', 'key', 'verb', 'worker_id',
       'assign_id', 'is_verbal', 'verb_form', 'question',
       'answer_range', 'answer', 'wh', 'subj', 'obj', 'obj2',
       'aux', 'prep', 'verb_prefix', 'is_passive', 'is_negated']
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
    :return:
    """
    from annotations_evaluations.decode_encode_answers import SPAN_SEPARATOR
    data_columns = ['qasrl_id', 'sentence', 'verb_idx', 'key', 'verb', 'verb_form']
    to_be_conjoined_columns = ['worker_id', 'assign_id']
    """ Rest of columns are annotation-columns - they are to be decoded from consolidated Response.
        (Except from 'answer' which requires both answer_range from Response and sentence.) """
    pred_dfs: List[pd.DataFrame] = []
    for key, pred_df in df.groupby('key'):
        responses = {worker: annotations_evaluations.decode_encode_answers.decode_response(worker_df)
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


def auto_consolidate_predicate(responses: List[Response]) -> Response:
    verb_form = responses[0].verb_form
    conjuncted_is_verbal = all(map(lambda r: r.is_verbal, responses))
    if not conjuncted_is_verbal:
        return Response(conjuncted_is_verbal, verb_form)

    # assuming Iterable of 2 responses
    resp1, resp2 = responses[:2]
    from annotations_evaluations.evaluate import find_matches
    alignment : Dict[Argument, Argument] = find_matches(resp1.all_args(), resp2.all_args())
    # take only aligned arguments along with their full Role (=QA)
    intersecting_roles : List[Role] = []
    # Helper func:
    def orig_role(arg: Argument, response: Response) -> Role:
        """ get original role (question & argument-set) of arg. """
        roles_containing_arg = list(filter(lambda r: arg in r.arguments, response.roles))
        # shouldn't ever happen
        assert len(roles_containing_arg)==1, f"More or less than one role contains this arg: {arg}, roles: {roles_containing_arg}"
        return roles_containing_arg[0]

    def is_conflicting(role: Role, pre_selected_roles: Set[Role]) -> bool:
        """ Return True if any of role.arguments if conflicting (overlapping) with the pre-selected arguments. """
        overlapping = lambda a1,a2: iou(a1,a2)>0
        return any(overlapping(arg, pre_arg)
                   for arg in role.arguments
                   for pre_role in pre_selected_roles
                   for pre_arg in pre_role.arguments)

    selected_roles : Set[Role] = set()
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
    return Response(conjuncted_is_verbal, verb_form, list(selected_roles))
