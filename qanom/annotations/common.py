from typing import NoReturn, Dict, List

import numpy as np
import pandas as pd


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
    from annotations.decode_encode_answers import decode_qasrl
    df = read_csv(file_path)
    return decode_qasrl(df)


def save_annot_csv(annot_df: pd.DataFrame, file_path: str) -> NoReturn:
    from annotations.decode_encode_answers import encode_qasrl
    df = encode_qasrl(annot_df)
    df.to_csv(file_path, index=False, encoding="utf-8")


""" Helper funcs for important information within an annotation DataFrame """

def set_key_column(annot_df: pd.DataFrame):
    """ Add 'key' column (predicate unique identifier) """
    if 'key' not in annot_df.columns:
        annot_df['key'] = annot_df.apply(lambda r: r['qasrl_id']+"_"+str(r['verb_idx']), axis=1)


def get_sent_map(annot_df: pd.DataFrame) -> Dict[str, List[str]]:
    sent_map = dict(zip(annot_df.qasrl_id, annot_df.sentence.apply(str.split)))
    return sent_map


def set_n_workers(df: pd.DataFrame) -> pd.DataFrame:
    # per predicate
    cols = ['qasrl_id', 'verb_idx']
    df['n_workers'] = df.groupby(cols).worker_id.transform(pd.Series.nunique)
    return df


def set_n_roles(df: pd.DataFrame) -> pd.DataFrame:
    # per predicate per worker
    cols = ['qasrl_id', 'verb_idx']
    df['n_roles'] = df.groupby(cols + ['worker_id']).verb.transform(pd.Series.count)
    return df


def get_n_predicates(df: pd.DataFrame) -> int:
    # overall
    cols = ['qasrl_id', 'verb_idx']
    return df[cols].drop_duplicates().shape[0]


def get_n_positive_predicates(worker_df: pd.DataFrame) -> int:
    # gets a df of a single worker, returns number of isVerbal==True in his annotations
    reduced_df = worker_df.drop_duplicates(subset=["key"])
    n_positive_predicates = reduced_df.is_verbal.sum()
    return n_positive_predicates