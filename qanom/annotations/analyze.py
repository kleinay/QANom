from collections import Counter
from typing import *

import pandas as pd

import qanom.evaluation.evaluate_inter_annotator as eia
from qanom import utils
from qanom.annotations.common import get_n_assignments, get_n_predicates, get_n_positive_predicates, filter_questions, \
    get_n_QAs, get_predicate_idx_label, get_n_args
from qanom.annotations.decode_encode_answers import Question, question_from_row


def isVerbalSumSeries(annot_df: pd.DataFrame) -> pd.Series:
    """
    in the returned Series is_verbal_sum, a key (= qasrl_id + _ + target_idx) is mapped to the number of generators
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
    from qanom.annotations.common import normalize, get_sent_map
    sent_map: Dict[str, List[str]] = get_sent_map(annot_df)
    cols = ['qasrl_id', 'target_idx', 'verb']
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


def print_annot_statistics(annot_df: pd.DataFrame):
    n_assignments = get_n_assignments(annot_df)
    n_predicates = get_n_predicates(annot_df)
    n_positive_predicates = get_n_positive_predicates(annot_df)
    n_qas = get_n_QAs(annot_df)
    annot_with_questions_df = filter_questions(annot_df)
    roleDist = Counter(annot_with_questions_df.groupby(['key','worker_id']).agg(pd.Series.count)['question'])
    sum_roles = sum(k * v for k, v in roleDist.items())
    num_roles_average = sum_roles / float(n_assignments)
    num_positive_wo_qas = roleDist[0]
    print(f'#-predicates: {n_predicates}')
    print(f'#-positive-predicates: {n_positive_predicates} (%{100*n_positive_predicates/float(n_predicates):.1f})')
    print(f'#-QAs (total): {n_qas}')
    print(f'#-Roles per predicate Distribution: {roleDist}')
    print(f'#-Roles average (for positive predicates): {num_roles_average:.2f}')
    print(f'#-positive predicates with NO role: {num_positive_wo_qas}'
          f'  ({num_positive_wo_qas/float(n_positive_predicates):.2f}% of positives)')


def compute_num_self_cycles(annot_df: pd.DataFrame) -> int:
    """ How many arguments in annot_df include the predicate within the answer span. """
    counter = 0
    idx_lbl = get_predicate_idx_label(annot_df)
    for i,row in annot_df.iterrows():
        answers = row.answer_range
        pred_idx = row[idx_lbl]
        for a in answers:
            if pred_idx in range(a[0], a[1]):
                counter += 1
    return counter


def compute_num_self_cycles_percentage(annot_df: pd.DataFrame) -> float:
    """ Num of predicate-including args / num-args"""
    n_self_loops = compute_num_self_cycles(annot_df)
    n_args = get_n_args(annot_df)
    return float(n_self_loops) / n_args


def analyze_question_feature_distribution(annot_df: pd.DataFrame, question_func: Callable[[Question,], Any]) -> Counter:
    """ given a function f mapping Questions to any value, return the distribution of f(q) for all Q in dataset """
    values = []
    for i,row in filter_questions(annot_df).iterrows():
        question: Question = question_from_row(row)
        values.append(question_func(question))
    return Counter(values)


def question_role_distribution(annot_df: pd.DataFrame) -> Counter:
    from qanom.evaluation.roles import question_to_sem_role
    rawCounter = analyze_question_feature_distribution(annot_df, question_func=question_to_sem_role)
    return utils.asRelative(rawCounter)