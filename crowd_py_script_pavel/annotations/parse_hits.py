# %load "../annotations/parse_hits.py"
import pandas as pd
import os
import codecs
from datetime import datetime, timedelta
from logging import log, ERROR, INFO
import json
from glob import glob
from tqdm import tqdm, tqdm_notebook
from stanfordcorenlp import StanfordCoreNLP
import shutil

from annotations.decode_encode_answers import encode_qasrl, is_invalid_range, NO_RANGE, encode_argument_text


def json_load(file_path):
    if not os.path.exists(file_path):
        log(ERROR, file_path)
        return None
    with codecs.open(file_path, "r", encoding="utf-8") as fd:
        return json.load(fd)

def parse_assign_time(assign):
    acc_time = int(assign['acceptTime'])
    acc_time = datetime.fromtimestamp(acc_time/1000.0)
    sub_time = int(assign['submitTime'])
    sub_time = datetime.fromtimestamp(sub_time/1000.0)
    assign_time = sub_time - acc_time
    return assign_time


def yield_parse_hits(hit_type_dir, parse_assignment):
    hit_dirs = glob(hit_type_dir + "/*")
    for hit_dir in hit_dirs:
        assign_files = glob(hit_dir + "/*.json")
        assign_files = [assign_file for assign_file in assign_files
                        if "hit.json" not in assign_file]
        hit_file = os.path.join(hit_dir, "hit.json")

        hit_json = json_load(hit_file)
        assign_jsons = [json_load(assign_file) for assign_file in assign_files]
        for assign_json in assign_jsons:
            yield parse_assignment(hit_json, assign_json)


def group_answers(hit_df):
    non_answer_cols = list(set(hit_df.columns.values) - {'answer_range'})
    hit_df2 = hit_df.groupby(non_answer_cols).answer_range.apply(pd.Series.tolist).reset_index()
    return hit_df2


CONCAT_THRESH = 3


def concat_answer_ranges(ranges):
    if ranges[0] == "NO_RANGE":
        return ranges

    ranges = sorted(ranges, key=lambda rng: rng[0])
    new_ranges = ranges
    if len(ranges) < CONCAT_THRESH:
        return new_ranges
    # we store ranges as strings, since other than here, they are not of much use
    # and better serialize them in one big string


    new_ranges = []
    prev_rng = ranges[0]
    for curr_rng in ranges[1:]:
        if prev_rng[1] == curr_rng[0]:
            # end of previous range is at start of current pair
            prev_rng = prev_rng[0], curr_rng[1]
        else:
            # this is a new start!
            new_ranges.append(prev_rng)
            prev_rng = curr_rng
    # OK reached end, add the last prev_rng
    new_ranges.append(prev_rng)

    return new_ranges


def get_answer_spans(answer_ranges, tokenized_sentence):
    spans = [" ".join(tokenized_sentence[r[0]:r[1]]) for r in answer_ranges]
    return spans


def get_answer_spans_df(hit_df, sentence_map, sentence_id_field):
    hit_df.answer_range = hit_df.answer_range.apply(concat_answer_ranges)
    answer_series = []
    # dataframe.apply(get_answer_spans)
    # doesn't work here with a list as return value
    for _, row in hit_df.iterrows():
        if row.answer_range[0] == "NO_RANGE":
            answer_series.append(["INVALID"])
            continue
        tokens = sentence_map[row[sentence_id_field]]
        answer_spans = get_answer_spans(row.answer_range, tokens)
        answer_series.append(answer_spans)

    hit_df['answer'] = pd.Series(answer_series)
    return hit_df


def get_verb(row, sentence_map, sentence_id_field):
    sentence = sentence_map[row[sentence_id_field]]
    return sentence[row.verb_idx]


def get_verb_df(hit_df, sentence_map, sentence_id_field):
    return hit_df.apply(
        get_verb,
        sentence_map=sentence_map,
        sentence_id_field=sentence_id_field,
        axis="columns")


### Parsing Question-Answer Generation Human Intelligence Task
# hit.json:
# ```javascript
# {
#     "hitTypeId": "3GTAJN9JK6ESH3VN7A09FC660C6XM3",
#     "hitId": "3EQVJH0T40MGJGD9HBEJEOLEHRPTHK",
#     "prompt": {
#         "id": { "index": 4 },
#         "verbIndex": 22
#     },
#     "creationTime": "1529249949352"
# }
# ```

#  [ASSIGNMENT_ID].txt:

# ```javascript
# {
#     "hitTypeId": "3GTAJN9JK6ESH3VN7A09FC660C6XM3",
#     "hitId": "3EQVJH0T40MGJGD9HBEJEOLEHRPTHK",
#     "assignmentId": "3T111IHZ5ERCAFFS5CQXPCHW5PSR9L",
#     "workerId": "A110KENBXU7SUJ",
#     "acceptTime": "1529250224000",
#     "submitTime": "1529250402000",
#     "response": [
#         {
#             "verbIndex": 22,
#             "question": "What was made to someone?",
#             "answers": [
#                 {
#                     "$type": "spacro.util.SpanImpl",
#                     "begin": 27,
#                     "end": 28
#                 }
#             ]
#         },
#         {
#             "verbIndex": 22,
#             "question": "Who was something made to?",
#             "answers": [
#                 {
#                     "$type": "spacro.util.SpanImpl",
#                     "begin": 23,
#                     "end": 24
#                 }
#             ]
#         },
#         {
#             "verbIndex": 22,
#             "question": "Who made something to someone?",
#             "answers": [
#                 {
#                     "$type": "spacro.util.SpanImpl",
#                     "begin": 17,
#                     "end": 18
#                 }
#             ]
#         }
#     ],
#     "feedback": ""
# }
# ```

def yield_qas(resp):
    for qa in resp:
        question = qa['question']
        for ans in qa['answers']:
            begin, end = ans['begin'], ans['end']+1
            answer_range = begin, end
            yield question, answer_range


def parse_generation_assignment_unvalidated(hit):
    prompt = hit['prompt']
    qasrl_id = prompt['genPrompt']['id']['id']
    verb_idx = prompt['genPrompt']['verbIndex']

    gens = [{
        "hit_id": prompt['sourceHITId'],
        "hit_type": prompt['sourceHITTypeId'],
        "assign_id": prompt['sourceAssignmentId'],
        "worker_id": "manual",
        "assign_time": 10,
        "qasrl_id": qasrl_id,
        "verb_idx": verb_idx,
        "question": question,
        "answer_range": answer_range
    } for question, answer_range in yield_qas(prompt['qaPairs'])]
    return pd.DataFrame.from_records(gens)


def parse_generation_assignment(hit, assign):
    assert(assign['hitId'] == hit['hitId'])
    qasrl_id = hit['prompt']['id']['id']

    verb_idx = hit['prompt']['verbIndex']

    assign_time = parse_assign_time(assign)

    for qa in assign['response']:
        assert(qa['verbIndex'] == verb_idx)
    gens = [{
        "hit_id": hit['hitId'],
        "hit_type": hit['hitTypeId'],
        "assign_id": assign['assignmentId'],
        "worker_id": assign['workerId'],
        "assign_time": assign_time,
        "qasrl_id": qasrl_id,
        "verb_idx": verb_idx,
        "question": question,
        "answer_range": answer_range
    } for question, answer_range in yield_qas(assign['response'])]

    return pd.DataFrame.from_records(gens)

# ### Parse Question-Answer Validation
# hit.json
# ```javascript
# {
#     "hitTypeId": "31L8DLROGDJJICGN7VLXX7OVWOX0AS",
#     "hitId": "3G5RUKN2EC4P9XQ6M34FS8HY9WV9N9",
#     "prompt": {
#         "genPrompt": {
#             "id": {"index": 1},
#             "verbIndex": 50
#         },
#         "sourceHITTypeId": "3ES7ZYWJEB6IMRJHPHMG4539HB6CH0",
#         "sourceHITId": "3JGHED38EDSF7D6RE8DCQ3E7B9UY7N",
#         "sourceAssignmentId": "3OONKJ5DKCKS1V29B4LWIYXHQFLBOY",
#         "qaPairs": [
#             {
#                 "verbIndex": 50,
#                 "question": "Who was going to do something?",
#                 "answers": [
#                     {
#                         "$type": "spacro.util.SpanImpl",
#                         "begin": 8,
#                         "end": 12
#                     },
#                     {
#                         "$type": "spacro.util.SpanImpl",
#                         "begin": 48,
#                         "end": 48
#                     }
#                 ]
#             },
#             {
#                 "verbIndex": 50,
#                 "question": "What was going to do something?",
#                 "answers": [
#                     {
#                         "$type": "spacro.util.SpanImpl",
#                         "begin": 51,
#                         "end": 55
#                     }
#                 ]
#             }
#         ]
#     },
#     "creationTime": "1529249951342"
# }
# ```

# [ASSIGNMENT_ID].json

# ```javascript
# {
#     "hitTypeId": "31L8DLROGDJJICGN7VLXX7OVWOX0AS",
#     "hitId": "3G5RUKN2EC4P9XQ6M34FS8HY9WV9N9",
#     "assignmentId": "3E1QT0TDFPAZ3BTZAXQABOLUYNS8IA",
#     "workerId": "A16F0CWJ0ZAR0B",
#     "acceptTime": "1529250072000",
#     "submitTime": "1529250208000",
#     "response": [
#         {
#             "$type": "qasrl.crowd.Answer",
#             "spans": [
#                 {
#                     "$type": "spacro.util.SpanImpl",
#                     "begin": 0,
#                     "end": 1
#                 }
#             ]
#         },
#         {
#             "$type": "qasrl.crowd.Answer",
#             "spans": [
#                 {
#                     "$type": "spacro.util.SpanImpl",
#                     "begin": 18,
#                     "end": 19
#                 }
#             ]
#         }
#     ],
#     "feedback": ""
# }
# ```


def yield_qa_validations(resp, quests):
    # verify validator has answered all questions
    assert (len(resp) == len(quests))
    for q, a in zip(quests, resp):
        if a['$type'] == "qasrl.crowd.Answer":
            for span in a['spans']:
                begin, end = span['begin'], span['end']+1
                ans_range = begin, end
                yield q, ans_range
        else:
            yield q, "NO_RANGE"


def parse_validation_assignment(hit, assign):
    assert(assign['hitId'] == hit['hitId'])
    qasrl_id = hit["prompt"]["genPrompt"]["id"]["id"]
    verb_idx = hit["prompt"]["genPrompt"]["verbIndex"]
    for q in hit["prompt"]['qaPairs']:
        assert(q['verbIndex'] == verb_idx)

    quests = [q["question"] for q in hit["prompt"]['qaPairs']]
    resp = assign['response']
    assign_time = parse_assign_time(assign)
    qas = [{
        "hit_id": hit["hitId"], "hit_type": hit['hitTypeId'],
        "assign_id": assign['assignmentId'], "worker_id": assign['workerId'],
        "assign_time": assign_time,
        "source_hit_id": hit["prompt"]['sourceHITId'],
        "source_hit_type": hit["prompt"]['sourceHITTypeId'],
        "source_assign_id": hit["prompt"]['sourceAssignmentId'],
        "qasrl_id": qasrl_id, "verb_idx": verb_idx,
        "question": question, "answer_range": ans_range
    } for question, ans_range in yield_qa_validations(resp, quests)]
    return pd.DataFrame.from_records(qas)


def get_validations(val_hit_dir, sentence_map, sentence_id_field):
    val_assignment_dfs = tqdm(yield_parse_hits(val_hit_dir, parse_validation_assignment))
    val_df = pd.concat(val_assignment_dfs)

    val_df2 = group_answers(val_df)
    val_df2 = get_answer_spans_df(val_df2, sentence_map, sentence_id_field)
    val_df2['verb'] = get_verb_df(val_df2, sentence_map, sentence_id_field)
    cols = ["qasrl_id", "verb_idx", "source_assign_id", "question"]
    val_df2['val_idx'] = val_df2.groupby(cols).worker_id.transform(pd.Series.rank)

    # val_cols = ['question', 'answer', 'verb', "qasrl_id",
    #             'verb_idx', 'answer_range', 'hit_id', 'assign_id', 'worker_id',
    #             'source_assign_id', 'hit_type', 'assign_time']

    val_cols = ["qasrl_id", 'verb_idx', 'source_assign_id', 'question', 'answer', 'verb', 'answer_range']
    val_df2 = val_df2[val_cols].copy()
    val_df2.sort_values(['qasrl_id', 'verb_idx', 'source_assign_id', 'question'], inplace=True)
    return val_df2


def get_generations(gen_hit_dir, sentence_map, sentence_id_field):
    gen_assignment_dfs = tqdm(yield_parse_hits(gen_hit_dir, parse_generation_assignment))
    gen_df = pd.concat(gen_assignment_dfs)

    gen_df2 = group_answers(gen_df)
    gen_df2 = get_answer_spans_df(gen_df2, sentence_map, sentence_id_field)
    gen_df2['verb'] = get_verb_df(gen_df2, sentence_map, sentence_id_field)

    gen_cols = ['question', 'answer', 'verb', "qasrl_id",
                'verb_idx', 'answer_range', 'worker_id', 'assign_id', 'hit_id',
                'hit_type', 'assign_time']

    gen_df2 = gen_df2[gen_cols].copy()
    return gen_df2


def combine_validated_answers(qasrl):
    cols = ['qasrl_id', 'verb_idx', 'source_assign_id', 'question']
    ans_cols = ['answer_range', 'answer', 'worker_id_val', 'assign_id']
    qasrl.sort_values(cols, inplace=True)
    val_indices = qasrl.groupby(cols).worker_id_val.transform(pd.Series.rank)
    val_indices = val_indices.astype(int)
    base_df = qasrl[cols + ['verb', 'worker_id_gen']].drop_duplicates()
    for val_idx in sorted(val_indices.unique()):
        ans_df = qasrl[val_indices == val_idx][cols + ans_cols].copy()

        ans_df.rename(columns={col_name: "{}_{}".format(col_name, val_idx)
                               for col_name in ans_cols}, inplace=True)
        base_df = pd.merge(base_df, ans_df, on=cols)
    base_df.sort_values(cols, inplace=True)
    return base_df


def translate_argument_to_text(row, sentence_map, argument_field):
    tokens = sentence_map[row.ecb_id]
    arg = row[argument_field]
    if arg[0] == NO_RANGE:
        return NO_RANGE

    arg_text = [" ".join(tokens[span[0]: span[1]]) for span in arg]
    return encode_argument_text(arg_text)



def main():
    # ROOT = os.path.join(".", "curated")
    ROOT = os.path.join(".", "mult_generation", "wikinews")
    SENTENCES_PATH = os.path.join(ROOT, "wikinews.dev.data3.csv")
    GEN_HIT_TYPE_ID = "wikinews.dev3.manual_annotated.2nd"
    VAL_HIT_TYPE_ID = "mult_curated"

    gen_hit_dir = os.path.join(ROOT, GEN_HIT_TYPE_ID)
    # val_hit_dir = os.path.join(ROOT, VAL_HIT_TYPE_ID)

    print(gen_hit_dir, os.path.exists(gen_hit_dir))
    # print(val_hit_dir, os.path.exists(val_hit_dir))

    sent_df = pd.read_csv(SENTENCES_PATH)
    sent_map = dict(zip(sent_df.qasrl_id, sent_df.tokens.apply(lambda t: t.split())))

    txt_files = glob(gen_hit_dir + "/**/*.txt")
    for txt_path in txt_files:
        print(txt_path)
        new_path = os.path.splitext(txt_path)[0] + ".json"
        shutil.move(txt_path, new_path)

    gen_df = get_generations(gen_hit_dir, sent_map, "qasrl_id")
    # val_df2 = get_validations(val_hit_dir, sent_map, "qasrl_id")

    # Only for curated dataset, where invalid really means question is either redundant or wrong
    # val_df2 = val_df2[~val_df2.answer_range.apply(is_invalid_range)].copy()

    out_path = os.path.join(ROOT, 'wikinews.dev3.manual_annotated.2nd.csv')
    encode_qasrl(gen_df).to_csv(out_path, index=False, encoding="utf-8")


    # gen_df2 = get_generations(gen_hit_dir, sent_map, "qasrl_id")
    # gen_cols = ["qasrl_id", 'verb_idx', 'question', 'answer', 'verb', 'answer_range']
    # gen_df2 = gen_df2[gen_cols].copy()
    # gen_df2.sort_values(['qasrl_id', 'verb_idx', 'question'], inplace=True)
    # encode_qasrl(gen_df2).to_csv("wikinews.dev.manual_annotated.csv", index=False, encoding="utf-8")
    # DIFFERENT TOKENIZERS ARE THE DEVIL.
    # spacy tokenizer makes different decisions for hyp-hen words
    # basic nltk.word_tokenize is ok, but for cases where the sentence begins with a dash ' as in a quote.
    # CORE_NLP_DIR = "c:/dev/stanford-corenlp-full-2018-10-05"
    # ecb_file_path = os.path.join(
    #     "annotations",
    #     "ECBPlus.qasrl.batch_{}.csv".format(BATCH_NUMBER))
    #
    # gen_hit_dir = os.path.join(ROOT, GEN_HIT_TYPE_ID)
    # val_hit_dir = os.path.join(ROOT, VAL_HIT_TYPE_ID)

    # evl_hit_dir = os.path.join(ROOT, EVL_HIT_TYPE_ID)
    # nrl_evl_dir = os.path.join(ROOT, NRL_EVL_TYPE_ID)

    # ecb_id_2_tokens = get_sentence_map(CORE_NLP_DIR, ecb_file_path)

    # gen_df = get_generations(gen_hit_dir, sent_map, "ecb_id")
    # val_df = get_validations(val_hit_dir, sent_map, "ecb_id")
    # val_df = get_validations(val_hit_dir, ecb_id_2_tokens, "ecb_id")
    # evl_df = get_validations(evl_hit_dir, ecb_id_2_tokens, "ecb_id")
    # nrl_eval_df = get_validations(nrl_evl_dir, ecb_id_2_tokens, "ecb_id")

    # to_csv(gen_df, "generation_{}.csv")
    # to_csv(gen_df, "generation_{}.csv")
    # to_csv(val_df, "validation_{}.csv".format(VAL_HIT_TYPE_ID))
    # to_csv(evl_df, "validation_{}.csv".format(EVL_HIT_TYPE_ID))
    # to_csv(nrl_eval_df, "validation_{}.csv".format(NRL_EVL_TYPE_ID))


# def get_sentence_map(CORE_NLP_DIR, ecb_file_path):
#     ecb = pd.read_csv(ecb_file_path)
#     corenlp = StanfordCoreNLP(CORE_NLP_DIR)
#     ecb['tokens'] = ecb.sentence.apply(lambda s: corenlp.word_tokenize(s))
#     ecb['ecb_id'] = ecb.apply(lambda r: r.file_name + "_" + str(r.sent_id), axis="columns")
#     corenlp.close()
#     ecb_id_2_tokens = dict(zip(ecb.ecb_id, ecb.tokens))
#     return ecb_id_2_tokens


if __name__ == "__main__":
    main()