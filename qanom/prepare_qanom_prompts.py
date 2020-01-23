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
import os
import sys

# add project basic directory to sys.path, in order to refer qanom as a package from anywhere
project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_dir)

from typing import *

from nltk.parse import CoreNLPParser

from qanom import catvar as catvar_util
from qanom import wordnet_util
from qanom.annotations_evaluations.common import read_csv
from qanom.verb_to_nom import SuffixBasedNominalizationCandidates as VTN

""" Define which resources should be used (by default) for filtering nouns as candidate nominalizations. """
resources = {"wordnet": True,
             "catvar": True,
             "affixes_heuristic": True}

vtn = VTN()

"""
To run the CoreNLPParser on your machine (port 9000), pre-run the following command from the unzipped directory of the 
stanford-core-nlp project (see https://www.khalidalnajjar.com/setup-use-stanford-corenlp-server-python/ for instructions): 
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentim
ent" -port 9000 -timeout 30000
"""
pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')


# load data from corpus
def get_sentences_from_csv(csv_fn):
    """
    Load raw data (sentence with sentence-id) from simple csv format
    :param csv_fn: file name for the simple raw_data file (sentenceId, tokens, sentence)
    :return: dict {sentenceId : sentence}
    """
    df = read_csv(csv_fn)
    possible_labels_for_sent_id = {'sentenceId', 'qasrl_id'}
    sent_id_label = list(possible_labels_for_sent_id & set(df.columns))[0]
    df = df.set_index(sent_id_label)
    records = df[['sentence']].to_records()
    return dict(records)


def tokenize_and_pos_tag(sentence):
    """
    Apply CoreNLP POS-Tagger on sentences.
    CoreNLP is running on server, so use it to be compatible with the qasrl-crowdsourcing POS tagging
    :param sentence: unicode string
    :return:
     pos is [ (token, POS) for each token in sentence ]
     tok is [ tok for each token in sentence ]
     returning (tok, pos)
    """
    pos_tagged = list(pos_tagger.tag(sentence.split(" ")))
    tokenized = [e[0] for e in pos_tagged]
    return tokenized, pos_tagged


# get common nouns ("NN", "NNS")
def get_common_nouns(posTaggedSent: List[Tuple[str, str]]):
    """
    Filter and retrieve common-nouns from the sentence (to be candidates for nominaliation)
    :param posTaggedSent: unicode string
    :return: list of (index, token) tuples
    """
    COMMON_NOUNS_POS = ["NN", "NNS"]
    nouns = []
    for index, (token, pos) in enumerate(posTaggedSent):
        if pos in COMMON_NOUNS_POS:
            nouns.append((index, token))
    return nouns


# verbalize using both WordNet, CatVar and our in-house suffix-based heuristic (=VTN: Verb To possible Nominalizations)
def get_verb_forms_from_lexical_resources(nn,
                                          wordnet=True,
                                          catvar=True,
                                          affixes_heuristic=True,
                                          filter_distant=False):
    wordnet_verbs = wordnet_util.convert_pos_by_lemmas(nn, wordnet_util.WN_NOUN, wordnet_util.WN_VERB) \
        if wordnet else []
    # apply distant-verbs filtering on wordnet verbs
    if filter_distant:
        wordnet_verbs = list(filter(lambda v: filter_distant_verb_forms(v, nn), wordnet_verbs))

    catvar_verbs = catvar_util.catvariate(nn) if catvar else []
    affixes_heuristic_verbs = vtn.get_source_verbs(nn) if affixes_heuristic else []
    # sort by distance
    vrbs = catvar_verbs + wordnet_verbs + affixes_heuristic_verbs
    vrbs = [v for v, w in wordnet_util.results_by_edit_distance(nn, vrbs)]
    if vrbs:
        return vrbs, True
    else:
        return [nn], False


def filter_distant_verb_forms(verb_form, noun):
    """ Return False for a verb_form whose edit-distance from noun is too large
    (indicating it's probably not morphologically related). """
    edit_distance = wordnet_util.levenshteinDistance(verb_form, noun)
    short_size, long_size = min(len(verb_form),len(noun)), max(len(verb_form),len(noun))
    edit_distance_without_suffix = wordnet_util.levenshteinDistance(verb_form[:short_size], noun[:short_size])
    avg_size = (len(verb_form)+len(noun))/2.0
    num_chars_maybe_arbitrarily_identical = short_size/3
    if edit_distance_without_suffix > short_size - num_chars_maybe_arbitrarily_identical:
        print("filtered out: +'"+verb_form+"' (edit-distance: "+str(edit_distance)+") for the noun '"+noun+"'")
        return False
    else:
        return True


def get_candidate_nouns_from_raw_csv(csv_fn, **resources):
    """
    @:param csv_fn: csv file containing raw sentences
    Returns: list of candidate_info (dict)
    """
    sentences = get_sentences_from_csv(csv_fn)
    all_candidates = []
    for sid, sentence in sentences.items():
        # POS-tagging and tokenize are dependant, so doing both together
        tokenizedSent, posTaggedSent = tokenize_and_pos_tag(sentence)
        for idx, nn in get_common_nouns(posTaggedSent):
            verb_forms, is_had_verbs = get_verb_forms_from_lexical_resources(nn, **resources)
            # take only common nouns that have optional verb-forms as candidates:
            if is_had_verbs:
                candidate_info = {"sentenceId": sid,
                                  "tokSent" : tokenizedSent,
                                  "targetIdx": idx,
                                  "verbForms": verb_forms[:5]}

                all_candidates.append(candidate_info)
    return all_candidates


if __name__ == "__main__":
    """ Read from command line arguments. """
    if len(sys.argv) < 3:
        raise Exception("Missing command line arguments: 2 required - sentence.csv (input) and prompts.json (output) ")
    # assume last two are the required arguments
    sentences_csv_fn, prompts_json_fn = sys.argv[-2:]

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
    candidates = get_candidate_nouns_from_raw_csv(sentences_csv_fn, **resources)

    """ Write candidates information to a JSON file. """
    json.dump(candidates, open(prompts_json_fn, "w"))
