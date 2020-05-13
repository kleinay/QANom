import json
import os
from typing import Dict, Any, NoReturn, List

import pandas as pd

JSON_OBJ = Dict[str, Any]

from qanom.annotations.common import save_annot_csv
from qanom.annotations.decode_encode_answers import decode_qasrl


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

    from annotations.decode_encode_answers import Response, Role, Question, Argument, encode_response
    data_columns = ['qasrl_id', 'sentence', 'verb_idx', 'key', 'verb', 'verb_form']
    from annotations.consolidation import FINAL_COLUMNS

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

