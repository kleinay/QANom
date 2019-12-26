from collections import Counter

import pandas as pd

import qanom.annotations_evaluations.evaluate_inter_annotator as eia
from qanom import utils


def isVerbalStatistics(annot_df: pd.DataFrame):
    n_predicates = eia.get_n_predicates(annot_df)
    reduced_df = annot_df.drop_duplicates(subset=["key", "worker_id"])   # reduced has one row per predicate per worker
    is_verbal_sum = reduced_df.groupby("key").is_verbal.sum().astype(int)
    """
    in the Series is_verbal_sum, a key (qasrl_id+_=verb_idx) is mapped to the number of generators 
    for which isVerbal==True.
    """
    is_verbal_agreement_dist = utils.asRelative(Counter(is_verbal_sum))
    # for clarity
    utils.replaceKeys(is_verbal_agreement_dist,
                      inplace=True,
                      oldKeys2NewKeys={0: "Not Verbal",
                                       2: "Verbal",
                                       1: "Disagree"})
    return is_verbal_agreement_dist



