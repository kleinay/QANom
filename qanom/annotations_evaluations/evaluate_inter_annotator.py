import os
from argparse import ArgumentParser
from itertools import combinations
from typing import List, Dict

import numpy as np
import pandas as pd

from annotations_evaluations.argument_first_evaluation import eval_datasets
from annotations_evaluations.decode_encode_answers import decode_qasrl


def evaluate_generator_agreement(annot_df: pd.DataFrame, sent_map: Dict[str, List[str]]):
    cols = ['qasrl_id', 'verb_idx']
    n_gen = annot_df.groupby(cols).worker_id.transform(pd.Series.nunique)
    workers = annot_df.worker_id.unique().tolist()
    n_workers = len(workers)
    annot_df = annot_df[n_gen == n_workers].copy()
    print("n_workers: ", n_workers)
    print("n_predicates: ", annot_df[cols].drop_duplicates().shape[0])
    print(f"worker_1\tworker_2\tprec\trecall\tf1")

    f1s = []
    for w1, w2 in combinations(workers, r=2):
        w1_df = annot_df[annot_df.worker_id == w1].copy()
        w2_df = annot_df[annot_df.worker_id == w2].copy()
        arg_metrics, role_metrics, _ = eval_datasets(w1_df, w2_df, sent_map, allow_overlaps=False)
        print(f"{w1}\t{w2}\t{arg_metrics.prec()}\t{arg_metrics.recall()}\t{arg_metrics.f1()}")
        f1s.append(arg_metrics.f1())

    f1s = np.array(f1s)
    print(f1s.mean(), f1s.std())


def read_csv(file_path: str):
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="Latin-1")


def main(root_dir: str, dataset_name: str):
    sent_path = os.path.join(root_dir, f'{dataset_name}.csv')
    sent_df = read_csv(sent_path)
    sent_map = dict(zip(sent_df.qasrl_id, sent_df.tokens.apply(str.split)))
    # original annotations, multiple generation tasks per predicate
    annot_df = read_csv(os.path.join(root_dir, f'{dataset_name}.annot.csv'))
    annot_df = decode_qasrl(annot_df)
    print(annot_df.worker_id.value_counts())
    evaluate_generator_agreement(annot_df, sent_map)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("inter_annotator_dir")
    ap.add_argument("dataset_name")
    args = ap.parse_args()
    main(args.inter_annotator_dir, args.dataset_name)

