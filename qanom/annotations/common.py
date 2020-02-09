from typing import NoReturn

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


def set_key_column(annot_df: pd.DataFrame):
    """ Add 'key' column (predicate unique identifier) """
    if 'key' not in annot_df.columns:
        annot_df['key'] = annot_df.apply(lambda r: r['qasrl_id']+"_"+str(r['verb_idx']), axis=1)
