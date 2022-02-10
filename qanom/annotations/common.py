from typing import NoReturn, Dict, List, Iterator, Tuple

import numpy as np
import pandas as pd

from qanom import utils
from qanom.annotations.decode_encode_answers import Response, decode_response


def normalize(lst):
    a = np.array(lst)
    return a / sum(a)


def read_dir_of_csv(dir_path: str, prefix="", suffix="", sep=',') -> pd.DataFrame:
    """ Concatenate (all) csv files in directory into one DataFrame """
    import os
    dfs, sections = zip(*[(read_csv(os.path.join(dir_path, fn), sep=sep), fn.rstrip(".csv"))
                          for fn in os.listdir(dir_path) if fn.endswith(suffix+".csv") and fn.startswith(prefix)])
    return pd.concat(dfs, ignore_index=True, keys=sections, sort=False)


def read_dir_of_annot_csv(dir_path: str, prefix="", suffix="") -> pd.DataFrame:
    """ Concatenate (all) csv files in directory into one DataFrame """
    import os
    dfs, sections = zip(*[(read_annot_csv(os.path.join(dir_path, fn)), fn.rstrip(".csv"))
                          for fn in os.listdir(dir_path) if fn.endswith(suffix+".csv") and fn.startswith(prefix)])
    return pd.concat(dfs, ignore_index=True, keys=sections, sort=False)


def read_csv(file_path: str, sep=',') -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep=sep)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="Latin-1", sep=sep)
    return df


def read_annot_csv(file_path: str) -> pd.DataFrame:
    from qanom.annotations.decode_encode_answers import decode_qasrl
    df = read_csv(file_path)
    return decode_qasrl(df)


def save_annot_csv(annot_df: pd.DataFrame, file_path: str) -> NoReturn:
    from qanom.annotations.decode_encode_answers import encode_qasrl
    df = encode_qasrl(annot_df)
    df.to_csv(file_path, index=False, encoding="utf-8")


def iterate_qanom_responses(annot_df: pd.DataFrame) -> Iterator[Tuple[str, str, int, str, Response]]:
    """
    An iterator over every QANomResponse in the annotation-DataFrame.
    :param annot_df: 
    :return: Iterator, every instance is (sent_id, sent, target_index, worker_id, Response)
    """
    pred_idx_label = get_predicate_idx_label(annot_df)
    for key, predicate_df in annot_df.groupby('key'):
        for worker_id, assignment_df in predicate_df.groupby('worker_id'):
            response = decode_response(assignment_df)
            row = assignment_df.iloc[0]
            sentence_id, sentence, target_index = row[["qasrl_id", "sentence", pred_idx_label]]
            yield sentence_id, sentence, target_index, worker_id, response


""" Helper funcs for important information within an annotation DataFrame """


def get_predicate_idx_label(df: pd.DataFrame) -> str:
    """ Return name of the column of the predicate-index (changed in final annotation files to 'target_idx') """
    return (set(df.columns) & {'verb_idx', 'target_idx'}).pop()


def get_predicate_str_label(df: pd.DataFrame) -> str:
    """ Return name of the column of the predicate-string (changed in final annotation files to 'noun') """
    return (set(df.columns) & {'verb', 'noun'}).pop()


def set_key_column(annot_df: pd.DataFrame):
    """ Add 'key' column (predicate unique identifier) """
    pred_idx_label = get_predicate_idx_label(annot_df)
    if 'key' not in annot_df.columns and len(annot_df):
        annot_df['key'] = annot_df.apply(lambda r: r['qasrl_id']+"_"+str(r[pred_idx_label]), axis=1)


def set_sentence_columns(annot_df: pd.DataFrame, sentence_df: pd.DataFrame) -> NoReturn:
    """ Set a 'sentence' column to `annot_df` based on its `qasrl_id`. Retrieve sentence from `sentence_df`."""
    sent_map = get_sent_map(sentence_df)
    annot_df['sentence'] = annot_df.apply(lambda r: ' '.join(sent_map[r.qasrl_id]), axis=1)


def get_sent_map(annot_df: pd.DataFrame) -> Dict[str, List[str]]:
    sent_map = dict(zip(annot_df.qasrl_id, annot_df.sentence.apply(str.split)))
    return sent_map


def set_n_workers(df: pd.DataFrame) -> pd.DataFrame:
    # per predicate
    df['n_workers'] = df.groupby('key').worker_id.transform(pd.Series.nunique)
    return df


def set_n_roles(df: pd.DataFrame) -> pd.DataFrame:
    # per predicate per worker
    df['no_roles'] = df.groupby(['key', 'worker_id']).wh.transform(lambda x: x.isnull().sum())
    df['num_rows'] = df.groupby(['key', 'worker_id']).wh.transform(pd.Series.count)
    df['n_roles'] = df.apply(lambda r: r['num_rows']-r['no_roles'], axis=1)
    df = df.drop('num_rows', axis=1)
    return df


def set_n_roles_per_predicate(df: pd.DataFrame) -> pd.DataFrame:
    # per predicate, count roles joint from all workers
    df['no_roles'] = df.groupby('key').wh.transform(lambda x: utils.is_empty_string_series(x).sum())
    df['num_rows'] = df.groupby('key').wh.transform(pd.Series.count)
    df['n_roles'] = df.apply(lambda r: r['num_rows']-r['no_roles'], axis=1)
    df = df.drop(['num_rows'], axis=1)
    return df


def get_n_predicates(df: pd.DataFrame) -> int:
    # overall
    return df['key'].drop_duplicates().shape[0]


def get_n_assignments(df: pd.DataFrame) -> int:
    # get number of assignments captured within the CSV
    pred_idx_label = get_predicate_idx_label(df)
    return df[['qasrl_id', pred_idx_label, 'worker_id']].drop_duplicates().shape[0]


def get_n_positive_predicates(annot_df: pd.DataFrame) -> int:
    # gets a df of annotation, returns number of assignments where isVerbal==True.
    # for a single worker df, this is the number of positive predicates.
    # for a multi-worker annot-df, this should be divided by num of assignments to get the positive rate.
    reduced_df = annot_df.drop_duplicates(subset=["key", "worker_id"])
    n_positive_predicates = reduced_df.is_verbal.sum()
    return n_positive_predicates


def get_n_QAs(annot_df: pd.DataFrame) -> int:
    not_questions = utils.count_empty_string(annot_df.question)
    return annot_df.shape[0] - not_questions


def get_n_args(annot_df: pd.DataFrame) -> int:
    def count_non_empty(lst):
        return sum(1 for a in lst if a)
    return sum(annot_df.answer.apply(count_non_empty))


def filter_questions(annot_df: pd.DataFrame) -> pd.DataFrame:
    """ Return subset of `annot_df` with rows that corresponds to a non-empty question """
    with_q = annot_df[~utils.is_empty_string_series(annot_df.question)]
    return with_q


def filter_non_questions(annot_df: pd.DataFrame) -> pd.DataFrame:
    """ Return subset of `annot_df` with rows that corresponds to an empty question
    (which are either not-verbal or marked with "no-QA-applicable")"""
    from qanom import utils
    with_q = annot_df[utils.is_empty_string_series(annot_df.question)]
    return with_q


def get_n_argument_taking_predicates(annot_df: pd.DataFrame) -> int:
    with_q = filter_questions(annot_df)
    return get_n_positive_predicates(with_q)
