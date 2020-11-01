"""
Utils for converting between QANom CSV format (and similarly QASRL-GS format as in (Roit et. al., 2020))
and Large-Scale-QA-SRL jsonl format (https://github.com/uwnlp/qasrl-bank/blob/master/FORMAT.md).
"""
import json, os
from dataclasses import asdict
from typing import Dict, Any, NoReturn, List, Tuple

import numpy as np
import pandas as pd

JSON_OBJ = Dict[str, Any]

from qanom.annotations.common import save_annot_csv, read_annot_csv, get_predicate_idx_label
from qanom.annotations.decode_encode_answers import SPAN_SEPARATOR, decode_qasrl, Question, Argument, question_from_row
from qanom.utils import jsonl, df_first_row, replaceKeys, removeKeys


"""
Export jsonl format to csv format 
"""
def jsonl_file_to_csv(qasrl_v2_fn: str, dest_dir: str) -> NoReturn:
    with open(qasrl_v2_fn, encoding='latin-1') as f:
        annot_df = pd.concat([sentence_json_to_df(json.loads(jline))
                              for jline in f],
                             ignore_index=True, sort=False)
    # save df in destination
    orig_dir, orig_name = os.path.split(qasrl_v2_fn)
    new_name = '.'.join(orig_name.split('.')[:-1]) + ".csv"
    dest_fn = os.path.join(dest_dir, new_name)
    save_annot_csv(decode_qasrl(annot_df), dest_fn)


def sentence_json_to_df(qasrlv2: JSON_OBJ) -> pd.DataFrame:
    sent_id = qasrlv2['sentenceId']
    tok_sent = qasrlv2['sentenceTokens']
    sentence = ' '.join(tok_sent)

    from qanom.annotations.decode_encode_answers import Response, Role, Question, Argument, encode_response
    data_columns = ['qasrl_id', 'sentence', 'verb_idx', 'key', 'verb', 'verb_form']
    from qanom.annotations.consolidation import FINAL_COLUMNS

    def verb_entry_to_df(vEntry: JSON_OBJ) -> pd.DataFrame:
        verbIdx = vEntry['verbIndex']
        verb = tok_sent[verbIdx]
        qaJsons = list(vEntry['questionLabels'].values())
        generator_id = qaJsons[0]['questionSources'][0] if qaJsons else ""

        # construct Response. then use it
        def qEntry_to_role(qEntry: JSON_OBJ) -> Role:
            # first get Question
            qDict : JSON_OBJ = qEntry['questionSlots']
            replce_underscore = lambda s: "" if s=="_" else s
            qDict = {k:replce_underscore(v) for k,v in qDict.items()}
            qDict["verb_slot_inflection"] = qDict.pop('verb')
            qDict.update(is_negated=qEntry['isNegated'],
                         is_passive=qEntry['isPassive'],
                         text=qEntry['questionString'],
                         verb_prefix="")
            question = Question(**qDict)
            # get answers - insert all answer spans from all validators even though it is overlapping and redundant)
            answer_spans : List[Argument] = [tuple(span)
                                             for ansJdg in qEntry['answerJudgments'] if 'spans' in ansJdg
                                             for span in ansJdg['spans']]
            # answer spans in JSON format are Tuple[int, int], same as defined by us as Argument
            return Role(question=question, arguments=tuple(answer_spans))

        qas: List[Role] = [qEntry_to_role(qa) for qa in qaJsons]
        response = Response(is_verbal=True, verb_form=verb, roles=qas)
        pred_df = encode_response(response, sentence)
        other_data = {'qasrl_id': sent_id,
                      'sentence': sentence,
                      'verb_idx': verbIdx,
                      'key': f"{sent_id}_{verbIdx}",
                      'verb': verb,
                      'verb_form': "",
                      'worker_id': generator_id,
                      'assign_id': ""
                      }
        pred_df = pred_df.assign(**other_data)
        return pred_df

    # concat all predicates of sentence
    verb_dfs = [verb_entry_to_df(entry) for entry in qasrlv2['verbEntries'].values()]
    if verb_dfs:
        df = pd.concat(verb_dfs,
                       ignore_index=True, sort=False)
    else:
        df = pd.DataFrame(columns=FINAL_COLUMNS)
    return df[FINAL_COLUMNS]


"""
Export csv format to jsonl format
"""
def qanom_csv_file_to_jsonl(qanom_csv_fn: str, dest_dir: str) -> NoReturn:
    annot_df = read_annot_csv(qanom_csv_fn)
    sentences_dicts = (sentence_df_to_sentence_jsonl_dict(sentence_df)
                       for qasrl_id, sentence_df in annot_df.groupby('qasrl_id'))
    # save jsonl in destination
    orig_dir, orig_name = os.path.split(qanom_csv_fn)
    new_name = '.'.join(orig_name.split('.')[:-1]) + ".jsonl"
    dest_fn = os.path.join(dest_dir, new_name)
    jsonl.dump(sentences_dicts, open(dest_fn, "w"))


def sentence_df_to_sentence_jsonl_dict(sent_df: pd.DataFrame) -> JSON_OBJ:
    obj = {}
    row: pd.Series = df_first_row(sent_df)
    obj['sentenceId'] = row['qasrl_id']
    obj['sentenceTokens'] = row['sentence'].split(" ")
    obj['verbEntries'] = dict(predicate_df_to_verb_entry(predicate_df)
                              for _, predicate_df in sent_df.groupby('key'))
    return obj


def predicate_df_to_verb_entry(predicate_df: pd.DataFrame) -> Tuple[str, JSON_OBJ]:
    """ Returns (predicate_index, predicate_entry), where predicate_entry is a dict
        with the keys: 'verbIndex', 'verbInflectedForms', 'questionLabels'. """
    obj = {}
    row: pd.Series = df_first_row(predicate_df)
    pred_idx_lbl = get_predicate_idx_label(predicate_df)
    obj['verbIndex'] = int(row[pred_idx_lbl])
    # additional information, only relevant for QANom and not QA-SRL:
    #   'isVerbal' - whether the candidate predicate is indeed a verbal predicate
    #       (otherwise, no QAs are annotated).
    #   'verbForm' - what is the verbal form (infinitive) of the nominalization
    obj['isVerbal'] = row.is_verbal
    obj['verbForm'] = row.verb_form
    if row.is_verbal:
        obj['verbInflectedForms'] = {} # todo, based on obj['verbForm']
        question_entries = [question_row_to_question_label(r)
                            for _, r in predicate_df.iterrows()]
        obj['questionLabels'] = {question_entry["questionString"]: question_entry
                                 for question_entry in question_entries}
    else:
        # negative instances (non verbal) would have empty questionLabels dict
        obj['verbInflectedForms'] = {}
        obj['questionLabels'] = {}
    return str(row[pred_idx_lbl]), obj


def question_row_to_question_label(row: pd.Series) -> JSON_OBJ:
    obj = {'questionString': row.question,
           'questionSources': [row.worker_id]
                                if 'source_worker_ids' not in row or np.isnan(row['source_worker_ids'])
                                else row['source_worker_ids'].split(SPAN_SEPARATOR),
           'isNegated': row.is_negated,
           'isPassive': row.is_passive,
           }
    # qasrl-v2 also have 'tense', 'isPerfect' and 'isProgressive' keys.
    # but currently it doesn't seem important to recover them from the qanom data.
    # if we would like to do that, the information is probably at 'verb_slot_inflection' column.
    question: Question = question_from_row(row)
    questionDict = asdict(question)
    replaceKeys(questionDict, {"verb_slot_inflection": "verb"})
    removeKeys(questionDict, ["text", "is_passive", "is_negated"])
    obj['questionSlots'] = questionDict
    # answers
    answer_spans: List[Argument] = row.answer_range
    # qanom or qasrl-gs data would only have a single answerJudgment because data is consolidated
    obj['answerJudgments'] = [{'sourceId': row.worker_id,
                               'isValid': True,
                               "spans": answer_spans}]
    return obj
