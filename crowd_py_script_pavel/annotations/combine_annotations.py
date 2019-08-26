import pandas as pd
import numpy as np
from typing import List, Tuple, Iterable
Span = Tuple[int, int]
Argument = List[Span]
ArgumentText = List[str]

from .decode_encode_answers import encode_argument_text, NO_RANGE

def combine_to_question_answer(question_df: pd.DataFrame, validation_dfs: List[pd.DataFrame]):
    common_fields = ['ecb_id','question', 'verb_idx', 'verb']
    full_val_df = pd.concat(validation_dfs)
    qasrl = pd.merge(question_df, full_val_df,
                     left_on=common_fields+['assign_id'],
                     right_on=common_fields+['source_assign_id'],
                     suffixes=["_gen", "_val"])

    qasrl['answer_idx'] = qasrl.groupby(common_fields).assign_id_val.transform(pd.Series.rank)
    qasrl['is_valid'] = qasrl.answer.apply(lambda a: a[0] != 'INVALID')
    return qasrl

def flatten_qasrl(qasrl_df: pd.DataFrame):
    common_fields = ['ecb_id','question', 'verb_idx', 'verb']
    # the base data frame, upon which we build all validator answers
    qasrl2 = qasrl_df[common_fields].drop_duplicates()
    n_answers_per_question = qasrl_df.answer_idx.nunique()
    for idx in range(1, n_answers_per_question + 1):
        candidate_df = qasrl_df[qasrl_df.answer_idx == idx].copy()
        candidate_df = candidate_df[common_fields + ['answer', 'answer_range', 'worker_id_val']].copy()
        candidate_df.rename(columns={'answer': 'answer_{}'.format(idx),
                                     'answer_range': 'answer_range_{}'.format(idx),
                                     'worker_id_val': 'worker_id_{}'.format(idx)},
                            inplace=True)
        # expand qasrl2 with field_{i} fields on top
        qasrl2 = pd.merge(qasrl2, candidate_df, on=common_fields)

    answer_fiels = ['answer_{}'.format(i) for i in range(1,n_answers_per_question + 1)]
    answer_range_fiels = ['answer_range_{}'.format(i) for i in range(1,n_answers_per_question + 1)]
    worker_fields = ['worker_id_{}'.format(i) for i in range(1,n_answers_per_question + 1)]
    qasrl2 = qasrl2[common_fields + answer_fiels + worker_fields + answer_range_fiels].copy()
    return qasrl2


