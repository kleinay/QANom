from collections import Counter
from typing import *

import pandas as pd

import annotations.common
import evaluation.evaluate_inter_annotator as eia
from qanom import utils


def isVerbalSumSeries(annot_df: pd.DataFrame) -> pd.Series:
    """
    in the returned Series is_verbal_sum, a key (= qasrl_id + _ + verb_idx) is mapped to the number of generators
    for which isVerbal==True.
    """
    reduced_df = annot_df.drop_duplicates(subset=["key", "worker_id"])   # reduced has one row per predicate per worker
    is_verbal_sum = reduced_df.groupby("key").is_verbal.sum().astype(int)
    return is_verbal_sum

def isVerbalStatistics(annot_df: pd.DataFrame):
    is_verbal_sum = isVerbalSumSeries(annot_df)
    is_verbal_agreement_dist = utils.asRelative(Counter(is_verbal_sum))
    # for clarity
    utils.replaceKeys(is_verbal_agreement_dist,
                      inplace=True,
                      oldKeys2NewKeys={0: "Not Verbal",
                                       2: "Verbal",
                                       1: "Disagree"})
    return is_verbal_agreement_dist


def most_controversial_predicates(annot_df: pd.DataFrame):
    from scipy import stats
    from qanom.annotations.common import normalize
    sent_map: Dict[str, List[str]] = annotations.common.get_sent_map(annot_df)
    cols = ['qasrl_id', 'verb_idx', 'verb']
    entropies = annot_df.groupby(cols).is_verbal.agg(lambda s: stats.entropy(normalize(s.value_counts()), base=2))
    print(entropies.sort_values(ascending=False))

    #todo compete it- compress to is_verbal decision per se, and see number of annotators



"""
It may be useful to show some of the quality metrics (e.g. IAA) on different splits of the dataset, for example, for
different domains, or different kinds of questions. Here are a bunch of helper funcs to do that.
"""
def analyze_by_column_value(annot_df: pd.DataFrame,
                            column: str,
                            analysis_func: Callable[[pd.DataFrame,],Any] = eia.evaluate_inter_generator_agreement):
    for key, df in annot_df.groupby(column):
        print(f"Results for {column}=={key}:")
        res = analysis_func(df)
        print(res or "")

def analyze_by_wh_word(annot_df: pd.DataFrame,
                       analysis_func: Callable[[pd.DataFrame,],Any] = eia.evaluate_inter_generator_agreement):
    analyze_by_column_value(annot_df, 'wh', analysis_func)

def analyze_by_wh_group(annot_df: pd.DataFrame,
                        analysis_func: Callable[[pd.DataFrame,],Any] = eia.evaluate_inter_generator_agreement,
                        wh_groups={"modifiers (where, when)": ["where", "when", "how long", "how much"],
                                   "core (who, what)": ["who", "what"],
                                   "implied (how, why)": ["how", "why"]}) -> Any:
    wh2groupName = {wh: grp
                    for grp, whs in wh_groups.items()
                    for wh in whs}
    # create a new column for group, to utilize analyze_by_column_value
    grouping_column = 'wh-group'
    annot_df[grouping_column]=annot_df.wh.apply(wh2groupName.get)
    return analyze_by_column_value(annot_df, column=grouping_column, analysis_func=analysis_func)

