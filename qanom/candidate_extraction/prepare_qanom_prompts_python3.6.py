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

from typing import List, Tuple, Iterable, Dict, Any

from nltk.parse import CoreNLPParser
import pandas as pd

from qanom.candidate_extraction import wordnet_util, catvar as catvar_util
from qanom.annotations.common import read_csv
from qanom.candidate_extraction.verb_to_nom import SuffixBasedNominalizationCandidates as VTN

""" Define which resources should be used (by default) for filtering nouns as candidate nominalizations. """
resources = {"wordnet": True,
             "catvar": True,
             "affixes_heuristic": True}

vtn = VTN()

"""
To run the CoreNLPParser on your machine (port 9000), pre-run the following command from the unzipped directory of the 
stanford-core-nlp project (see https://www.khalidalnajjar.com/setup-use-stanford-corenlp-server-python/ for instructions): 
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000
"""
pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
pos_tag = pos_tagger.tag

"""
Alternatively- use nltk's default pos_tagger ('averaged_perceptron_tagger'):

nltk.download('averaged_perceptron_tagger')
pos_tag = nltk.pos_tag
"""

COMMON_NOUNS_POS = ["NN", "NNS"]


# load data from corpus
def get_sentences_from_csv(csv_fn):
    """
    Load raw data (sentence with sentence-id) from simple csv format
    :param csv_fn: file name for the simple raw_data file (sentenceId, sentence)
    :return: dict {sentenceId : sentence}
    """
    df = read_csv(csv_fn)
    possible_labels_for_sent_id = {'sentenceId', 'sentence_id', 'qasrl_id'}
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
    pos_tagged = list(pos_tag(sentence.split(" ")))
    tokenized = [e[0] for e in pos_tagged]
    return tokenized, pos_tagged


# get common nouns ("NN", "NNS")
def get_common_nouns(posTaggedSent: List[Tuple[str, str]]):
    """
    Filter and retrieve common-nouns from the sentence (to be candidates for nominaliation)
    :param posTaggedSent: [ (token1, pos1), ...]
    :return: list of (index, token) tuples
    """
    nouns = []
    for index, (token, pos) in enumerate(posTaggedSent):
        if pos in COMMON_NOUNS_POS:
            nouns.append((index, token))
    return nouns


# verbalize using both WordNet, CatVar and our in-house suffix-based
# heuristic (=VTN: Verb To possible Nominalizations; implementation at `verb_to_nom.py`)
def get_verb_forms_from_lexical_resources(nn,
                                          wordnet=True,
                                          catvar=True,
                                          affixes_heuristic=True,
                                          filter_distant=False) -> Tuple[List[str], bool]:
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


def is_candidate_noun(nn: str, **resources) -> bool:
    # does it have candidate (morphologically related) verb-forms?
    return get_verb_forms_from_lexical_resources(nn, **resources)[1]


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

"""
Different functions for different input formats of sentences.
"""
def get_candidate_nouns_from_allennlp_jsonl(sentences_josnl_fn: str, **resources) -> List[Dict[str, Any]]:
    """
    @:param sentences_josnl_fn: a file name of JSON-lines of input sentences, as allennlp predictors input format:
        {"sentence": <actual sentence string> }
        {"sentence": <actual sentence string> }
        ...
    Returns: list of candidate_info (dict). Generated a unique sentence_id for the sentences (by index).
    """
    with open(sentences_josnl_fn) as f:
        sentences = []
        for line in f.readlines():
            sentences.append(json.loads(line)["sentence"])
    return get_candidate_nouns_from_raw_sentences(sentences, **resources)


def get_candidate_nouns_from_raw_sentences(sentences: Iterable[str], **resources) -> List[Dict[str, Any]]:
    """
    @:param sentences: iterable of raw sentences
    Returns: list of candidate_info (dict). Generated a unique sentence_id for the sentences (by index).
    """
    sentences_dict: Dict[str, str] = {f"Sentence-{1+i:04}": sent for i, sent in enumerate(sentences)}
    return get_candidate_nouns(sentences_dict, **resources)


def get_candidate_nouns_from_raw_csv(csv_fn, **resources) -> List[Dict[str, Any]]:
    """
    @:param csv_fn: csv file containing raw sentences
    Returns: list of candidate_info (dict)
    """
    sentences: Dict[str, str] = get_sentences_from_csv(csv_fn)
    return get_candidate_nouns(sentences, **resources)

# the "core" function of extracting candidates from sentences
def get_candidate_nouns(sentences: Dict[str, str], **resources) -> List[Dict[str, Any]]:
    """
    @:param sentences: {sentence_id : sentence_string}
    Returns: list of candidate_info (dict)
    """
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


def export_candidate_info_to_csv(candidates_info: List[Dict[str, Any]], csv_out_fn: str = None) -> pd.DataFrame:
    """
    Export the noun-candidates info (returned by `get_candidate_nouns`) into a QANom-csv format,
     with columns: 'qasrl_id', 'sentence', 'target_idx', 'noun'.
    It is useful when you want to use a candidate extraction heuristic as a pre-processing for QANom model
    that was trained on QANom data (which is in qanom csv format).

    :param candidates_info: see output of `get_candidate_nouns`
    :param csv_out_fn: [optional] a file name to write the csv into.
    :return: a DataFrame of the produced CSV
    """
    df = pd.DataFrame(candidates_info)
    # now modify the dataFrame to match qanom annotation format
    df['sentence'] = df.apply(lambda r: ' '.join(r['tokSent']), axis=1)
    df['noun'] = df.apply(lambda r: r['tokSent'][int(r['targetIdx'])], axis=1)
    df = df.drop(['tokSent', 'verbForms'], axis='columns')
    df = df.rename(mapper={'targetIdx': 'target_idx',
                           'sentenceId': 'qasrl_id'},
                   axis='columns')
    if csv_out_fn:
        df.to_csv(csv_out_fn, index=False)
    return df


def extract_candidate_nouns_to_csv(input: Any, csv_out_fn: str,
                                   input_format):
    candidate_info_funcs = {"iterable": get_candidate_nouns_from_raw_sentences,
                            "dict": get_candidate_nouns,
                            "csv": get_candidate_nouns_from_raw_csv,
                            "jsonl": get_candidate_nouns_from_allennlp_jsonl}
    candidates_info = candidate_info_funcs[input_format](input)
    export_candidate_info_to_csv(candidates_info, csv_out_fn)


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
