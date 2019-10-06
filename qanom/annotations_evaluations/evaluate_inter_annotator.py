from argparse import ArgumentParser
from itertools import combinations
from typing import List, Dict

import numpy as np
import pandas as pd

from annotations_evaluations.argument_first_evaluation import eval_datasets
from annotations_evaluations.common import read_csv
from annotations_evaluations.decode_encode_answers import decode_qasrl


def evaluate_generator_agreement(annot_df: pd.DataFrame, sent_map: Dict[str, List[str]]):
    cols = ['qasrl_id', 'verb_idx']
    n_gen = annot_df.groupby(cols).worker_id.transform(pd.Series.nunique)
    workers = annot_df.worker_id.unique().tolist()
    n_workers = len(workers)
    annot_df = annot_df[n_gen == n_workers].copy()
    n_predicates = annot_df[cols].drop_duplicates().shape[0]
    print("n_workers: ", n_workers)
    print("n_predicates: ", n_predicates)
    print(f"metric\tworker_1\tworker_2\tprec\trecall\tf1")

    f1s = []
    for w1, w2 in combinations(workers, r=2):
        w1_df = annot_df[annot_df.worker_id == w1].copy()
        w2_df = annot_df[annot_df.worker_id == w2].copy()
        arg_metrics, role_metrics, nom_ident_metrics, _ = eval_datasets(w1_df, w2_df, sent_map, allow_overlaps=False)
        print(f"ARG:\t{w1}\t{w2}\t{arg_metrics.prec()}\t{arg_metrics.recall()}\t{arg_metrics.f1()}")
        print(f"ROLE:\t{w1}\t{w2}\t{role_metrics.prec()}\t{role_metrics.recall()}\t{role_metrics.f1()}")
        print(f"NOM_IDENT:\t{w1}\t{w2}\t{nom_ident_metrics.prec()}\t{nom_ident_metrics.recall()}\t{nom_ident_metrics.f1()}")
        print(f"NOM_IDENT accuracy: {nom_ident_metrics.accuracy(n_predicates)}%, {int(nom_ident_metrics.errors())} mismathces out of {n_predicates} predicates.")
        f1s.append(arg_metrics.f1())

    f1s = np.array(f1s)
    print(f1s.mean(), f1s.std())


def main(annotation_path: str, sentences_path: str = None):
    annot_df = read_csv(annotation_path)
    annot_df = decode_qasrl(annot_df)
    if sentences_path:
        sent_df = read_csv(sentences_path)
        sent_map = dict(zip(sent_df.qasrl_id, sent_df.tokens.apply(str.split)))
    else:
        sent_map = dict(zip(annot_df.qasrl_id, annot_df.sentence.apply(str.split)))
    # original annotations, multiple generation tasks per predicate
    print(annot_df.worker_id.value_counts())
    evaluate_generator_agreement(annot_df, sent_map)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("annotation_path")
    ap.add_argument("sentences_path", required=False)
    args = ap.parse_args()
    main(args.annotation_path, args.sentences_path)

