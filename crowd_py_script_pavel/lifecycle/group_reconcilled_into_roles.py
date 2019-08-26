import pandas as pd

from annotations.decode_encode_answers import decode_qasrl, SPAN_SEPARATOR, encode_qasrl
from annotations.parse_hits import group_answers


def paranthesis_to_encoded_span(s: str):
    begin, end = s.strip("()").split(",")
    return (int(begin), int(end))


reco_path = "wikinews.dev3.ground_truth_ver_1.csv"
df = pd.read_csv(reco_path)

# TODO FIX THIS AT SOURCE IN EVALUATION CODE
df.answer_range = df.answer_range.apply(paranthesis_to_encoded_span)

# from group_answers:
non_answer_cols = list(set(df.columns.values) - {'answer_range', 'answer'})
df2 = df.groupby(non_answer_cols).answer_range.apply(pd.Series.tolist).reset_index()

df3 = df.groupby(non_answer_cols).answer.apply(pd.Series.tolist).reset_index()
df_fixed = pd.merge(df2, df3, on=non_answer_cols)

encode_qasrl(df_fixed).to_csv("wikinews.dev3.ground_truth_ver_1.processed.csv", index=False, encoding="utf-8")