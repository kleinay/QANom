"""
This script can be used for generating a JSON file that serves as input to the qasrl-crowdsourcing MTurk application
for annotating verbal arguments of nominalizations (qanom-unified branch).

Input:
    arg1: a file name containing a CSV with a list of sentences (sentenceId, sentence)
    arg2: a file name for output JSON. The script will override this file and write the prompts to it in JSON format.

Logic building blocks:
* load raw data  (sentences) from corpus
* use Stanford CoreNLP to POS-tag the sentence
* get common nouns ("NN", "NNS")
* filter by wordnet + CatVar (+ possible-noms from the inhouse verb_to_nom utility)
* generate a list of prompts, each containing:
    { "sentenceId": sentence ID (string),
      "tokSent": sentence tokens (list),
      "targetIdx": index of candidate noun (int),
      "verbForms": non-empty list of verb-forms (strings) (ordered by edit distance)
    }

"""


import json
import sys

import prepare_nom_ident_batch

if __name__ == "__main__":
    """ Read from command line arguments. """
    if len(sys.argv) != 2:
        raise Exception("Missing command line arguments: 2 required - sentence.csv (input) and prompts.json (output) ")
    sentences_csv_fn, prompts_json_fn = sys.argv

    """ Define which resources should be used for filtering nouns as candidate nominalizations. """
    resources = {"wordnet" : True,
                 "catvar" : True,
                 "affixes_heuristic": True}

    """
    Use prepare_nom_ident_batch script to get candidates information as list of dicts.
    candidate_info = {"context": sentence,
                      "intent": idx,
                      "noun": tokenized_sent[idx],
                      "verb_forms": ' '.join(verb_forms[:5]),
                      "sentence_id": sid,
                      "is_factuality_event": "yes",
                      "is_had_auto_verbs": "yes" if is_had_verbs else "no"}
    """
    candidates_prev_format = prepare_nom_ident_batch.get_candidate_nouns_from_raw_csv(sentences_csv_fn, **resources)

    """ Change format of each candidate - remove redundant info, tokenize sentence, rename keys, etc. """
    candidates = [{"sentenceId": cdd["sentence_id"],
                   "tokSent" : cdd["context"].split(" "),
                   "targetIdx": cdd["intent"],
                   "verbForms": cdd["verb_forms"].split(" ")}
                  for cdd in candidates_prev_format]

    """ Write candidates information to a JSON file. """
    json.dump(candidates, open(prompts_json_fn, "w"))


