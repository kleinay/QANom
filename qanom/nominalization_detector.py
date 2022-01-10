"""
A unified class for detecting "verbal" nominalizations from raw text -
wraps both `candidate_extraction` and `predicate_detector`
"""
from typing import Iterable, List, Dict, Any
import sys, os
from transformers import AutoModelForTokenClassification, AutoTokenizer
import pandas as pd

import config
sys.path.append(os.path.dirname(config.qanom_package_root_path))
from qanom.candidate_extraction.candidate_extraction import extract_candidate_nouns

def dict_without(orig_dict: Dict[Any, Any], keys_to_remove: Iterable[Any]) -> Dict[Any, Any]:
    new_dict = orig_dict.copy()
    for key in keys_to_remove:
        if key in new_dict:
            new_dict.pop(key)
    return new_dict

class NominalizationDetector():    
    original_model_hub_name = "kleinay/nominalization-candidate-classifier"
    
    def __init__(self, model_hub_name = None):
        self.model_hub_name = model_hub_name or NominalizationDetector.original_model_hub_name
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_hub_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_hub_name)

    def __call__(self, raw_sentences: Iterable[str], 
                 return_all_candidates: bool = False, 
                 return_probability: bool = True,
                 threshold: float = 0.5):
        # use qanom's candidate_extraction module to identify candidates, i.e. nouns with morphologically-related verbs (using lexical resources)
        candidate_infos: List[Dict[str, Any]] = extract_candidate_nouns(raw_sentences, "iterable")
        # keys: 'sentenceId', 'tokSent', 'targetIdx', 'verbForms'
        cand_df = pd.DataFrame(candidate_infos)
        grouped_df = cand_df.groupby('sentenceId') 
        input_sent_df = pd.DataFrame()  # model is sentence level, so prepare input at sentence level
        input_sent_df["words"] = grouped_df.tokSent.first()
        input_sent_df["target_idxs"] = grouped_df.targetIdx.apply(list)
        input_sent_df["verb_forms"] = grouped_df.verbForms.apply(lambda r: [l[0] for l in r])

        assert len(raw_sentences) == len(input_sent_df)

        # just for reference:
        label_map = self.model.config.id2label
        positive_lbl_id = 0     # because label_map[0] == 'True' 
        negative_lbl_id = 1     # because label_map[1] == 'False' 
        
        tokenized = self.tokenizer(raw_sentences, return_tensors="pt", padding=True)
        logits = self.model(tokenized.input_ids).logits
        # compose binary probability as the mean of positive label's prob with complement of neg. label's prob.
        positive_probability = logits[:,:,positive_lbl_id].sigmoid()   # probability of the "true" label (index 0 in axis-2)
        negative_probability = logits[:,:,negative_lbl_id].sigmoid()
        probability = (positive_probability + (1-negative_probability) ) / 2
        preds = probability > threshold

        batch_size, seq_len = preds.shape # same shape as probability
        predicate_lists = [{} for _ in range(batch_size)]
        for i in range(batch_size):
            row = input_sent_df.iloc[i]
            # get mapping of word to tokens (for using indexes from dataset)
            wordId2numOfTokens = [len(self.tokenizer.tokenize(word)) for word in row.words]
            wordId2firstTokenId, curr_tok_id = [], 1
            for wordId in range(len(wordId2numOfTokens)):
                wordId2firstTokenId.append(curr_tok_id)
                curr_tok_id += wordId2numOfTokens[wordId]
            # get dict of all candidates to prediction (on first token of word)
            predicate_lists[i] = [{"predicate_idx": idx,
                                   "predicate": row.words[idx],
                                   "predicate_detector_prediction": preds[i][wordId2firstTokenId[idx]].item(),
                                   "predicate_detector_probability": probability[i][wordId2firstTokenId[idx]].item(),
                                   "verb_form": verb_form} 
                                 for idx, verb_form in zip(row.target_idxs, row.verb_forms)]

        # filter nominalizations that were positively classified as carrying a verbal meaning in context
        if not return_all_candidates:
            keys_to_remove = ["predicate_detector_prediction"] + ([] if return_probability else ["predicate_detector_probability"])
            for i, sent_predicate_list in enumerate(predicate_lists):
                predicate_lists[i] = [dict_without(predicate_dict, keys_to_remove) 
                                      for predicate_dict in sent_predicate_list
                                      if predicate_dict["predicate_detector_prediction"]]
        return predicate_lists

if __name__ == "__main__":
    raw_sentences: Iterable[str] = ["the construction of the officer 's building finished right after the beginning of the destruction of the previous construction ."]
    detector = NominalizationDetector()
    from pprint import pprint
    pprint(detector(raw_sentences, return_all_candidates=True))
    pprint(detector(raw_sentences, threshold=0.75, return_probability=False))
