"""
A unified class for detecting "verbal" nominalizations from raw text -
wraps both `candidate_extraction` and `predicate_detector`
"""
from typing import Iterable, List, Dict, Any, Tuple, Optional
from transformers import AutoModelForTokenClassification, AutoTokenizer
import pandas as pd

from qanom.candidate_extraction.candidate_extraction import get_candidate_nouns_from_pos_tagged_sentences, pos_tag_sentence

def dict_without(orig_dict: Dict[Any, Any], keys_to_remove: Iterable[Any]) -> Dict[Any, Any]:
    new_dict = orig_dict.copy()
    for key in keys_to_remove:
        if key in new_dict:
            new_dict.pop(key)
    return new_dict

class NominalizationDetector():    
    original_model_hub_name = "kleinay/nominalization-candidate-classifier"
    
    def __init__(self, model_hub_name = None, device: int = -1):
        "device (int, optional): -1 for CPU (default), >=0 refers to CUDA device ordinal. Defaults to -1."
        self.model_hub_name = model_hub_name or NominalizationDetector.original_model_hub_name
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_hub_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_hub_name)
        self.device = f"cuda:{device}" if device>=0 else "cpu" # represent device in pytorch convention 

    def __call__(self, raw_sentences: Iterable[str], 
                 return_all_candidates: bool = False, 
                 return_probability: bool = True,
                 threshold: float = 0.5,
                 pos_tagged_sentences: Optional[Iterable[List[Tuple[str, str]]]] = None):
        """
        tokens_pos: list of tuple s.t. tuple[0] = tokens sentence and tuple[1] = POS of sentence
        Returns a list (sized as `raw_sentences`) of lists of nominalization infos (dicts). 
        """
        # use qanom's candidate_extraction module to identify candidates, i.e. nouns with morphologically-related verbs (using lexical resources)
        # sentences_dict = {str(i): s for i,s in enumerate(raw_sentences)}

        if pos_tagged_sentences:
            pos_tagged_sentences_dict = {str(i): token_and_pos for i, token_and_pos in enumerate(pos_tagged_sentences)}
        else:
            pos_tagged_sentences_dict = {str(i): pos_tag_sentence(s) for i, s in enumerate(raw_sentences)}

        candidate_infos: List[Dict[str, Any]] = get_candidate_nouns_from_pos_tagged_sentences(pos_tagged_sentences_dict)
        # keys: 'sentenceId', 'tokSent', 'targetIdx', 'verbForms'
        
        # edge case - candidate_infos is totally empty
        if not candidate_infos:
            return [[]] * len(raw_sentences)
        cand_df = pd.DataFrame(candidate_infos)
        grouped_df = cand_df.groupby('sentenceId') 
        input_sent_df = pd.DataFrame()  # model is sentence level, so prepare input at sentence level
        input_sent_df["words"] = grouped_df.tokSent.first()
        input_sent_df["sent_idx"] = grouped_df.sentenceId.first().astype(int) #int(grouped_df.sentenceId.first())
        input_sent_df["target_idxs"] = grouped_df.targetIdx.apply(list)
        input_sent_df["verb_forms"] = grouped_df.verbForms.apply(lambda r: [l[0] for l in r])

        # since not all raw_sentences necessarily has candidates, must keep mapping to original sentence index
        included_sentences = [raw_sentences[i] for i in list(input_sent_df["sent_idx"])]
        included_idx2orig_idx = list(input_sent_df["sent_idx"])
        
        # just for reference:
        label_map = self.model.config.id2label
        positive_lbl_id = 0     # because label_map[0] == 'True' 
        negative_lbl_id = 1     # because label_map[1] == 'False' 
        
        tokenized = self.tokenizer(included_sentences, return_tensors="pt", padding=True).to(self.device)
        model = self.model.to(self.device)
        logits = model(tokenized.input_ids).logits
        # compose binary probability as the mean of positive label's prob with complement of neg. label's prob.
        positive_probability = logits[:,:,positive_lbl_id].sigmoid()   # probability of the "true" label (index 0 in axis-2)
        negative_probability = logits[:,:,negative_lbl_id].sigmoid()
        probability = (positive_probability + (1-negative_probability) ) / 2
        preds = probability > threshold

        batch_size, seq_len = preds.shape # same shape as probability
        assert batch_size == len(included_idx2orig_idx)
        
        predicate_lists = [[]] * len(raw_sentences) # final returned list is aligned with raw_sentences
        for i, orig_idx in enumerate(included_idx2orig_idx):
            # orig_idx is the sentence index in the input arg `raw_sentence`
            row = input_sent_df.iloc[i]
            # get mapping of word to tokens (for using indexes from dataset)
            wordId2numOfTokens = [len(self.tokenizer.tokenize(word)) for word in row.words]
            wordId2firstTokenId, curr_tok_id = [], 1
            for wordId in range(len(wordId2numOfTokens)):
                wordId2firstTokenId.append(curr_tok_id)
                curr_tok_id += wordId2numOfTokens[wordId]
            # get dict of all candidates to prediction (on first token of word)
            predicate_lists[orig_idx] = [
                {"predicate_idx": pred_idx,
                 "predicate": row.words[pred_idx],
                 "predicate_detector_prediction": preds[i][wordId2firstTokenId[pred_idx]].item(),
                 "predicate_detector_probability": probability[i][wordId2firstTokenId[pred_idx]].item(),
                 "verb_form": verb_form} 
                for pred_idx, verb_form in zip(row.target_idxs, row.verb_forms)]

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
