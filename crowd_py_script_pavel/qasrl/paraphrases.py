from collections import namedtuple

import pandas as pd
import numpy as np

QUESTION_FIELDS = ['wh', 'subj', 'obj', 'is_passive', 'is_negated']

Question = namedtuple('Question', ['text', 'gen_idx', 'wh', 'subj', 'obj', 'is_passive', 'is_negated'])


def is_paraphrase(q1: Question, q2: Question):
    if q1.text.lower() == q2.text.lower():
        return True

    if pd.isnull(q1.wh) or pd.isnull(q2.wh):
        return False

    eqs = [q1.wh == q2.wh,
           q1.subj == q2.subj,
           q1.obj == q2.obj,
           q1.is_passive == q2.is_passive,
           q1.is_negated == q2.is_negated]
    if all(eqs):
        print("found paraphrase ", q1.text, q2.text)

    return all(eqs)


def load_parsed_questions(questions_path):
    parsed_questions = pd.read_csv(questions_path)
    cols = ['qasrl_id', 'verb_idx', 'question', 'source_assign_id'] + QUESTION_FIELDS
    parsed_questions = parsed_questions[cols].copy()
    parsed_questions.is_negated = parsed_questions.is_negated.map(lambda neg: "NEG" if neg else "")
    parsed_questions.is_passive = parsed_questions.is_passive.map(lambda pas: "PASSIVE" if pas else "ACTIVE")
    parsed_questions.fillna("", inplace=True)
    return parsed_questions


def groupby_grammar(df):
    cols = ['wh', 'subj', 'obj', 'is_passive', 'is_negated', 'qasrl_id', 'verb_idx']
    group_idx = df.groupby(cols).ngroup()
    return group_idx


def main():
    df = load_parsed_questions("../mult_generation/wikinews/wikinews.dev.qasrl.mult_gen.parsed_questions.csv")
    df['group_idx'] = groupby_grammar(df)

    print(df.head(20))


if __name__ == "__main__":
    main()