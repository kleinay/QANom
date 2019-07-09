"""
This script can be used for generating a CSV file that serves as input to the MTurk nom-identification task.
Input: a list of sentences.

Logic building blocks:
* load raw data  (sentences) from corpus
* use Stanford CoreNLP to POS-tag the sentence
* get common nouns ("NN", "NNS")
* filter by wordnet + possible-nons
* generate prompts for identification task

"""

import nltk
import pandas as pd
from nltk.parse import CoreNLPParser

import catvar as catvar_util
import verb_to_nom
import wordnet_util

pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')


# load data from corpus
def get_sentences_from_csv(csv_fn):
    """
    Load raw data (sentence with sentence-id) from simple csv format
    :param csv_fn: file name for the simple raw_data file (sentenceId, tokens, sentence)
    :return: dict {sentenceId : sentence}
    """
    df = pd.read_csv(csv_fn)
    possible_labels_for_sent_id = {'sentenceId', 'qasrl_id'}
    sent_id_label = list(possible_labels_for_sent_id & set(df.columns))[0]
    df = df.set_index(sent_id_label)
    records = df[['sentence']].to_records()
    return dict(records)


def pos_tag(sentence):
    """
    Apply CoreNLP POS-Tagger on sentences.
    CoreNLP is running on server, so use it to be compatible with the qasrl-crowdsourcing POS tagging
    :param sentence: unicode string
    :return: { sentenceId : [ (token, POS) for each token in sentence ] }
    """
    return list(pos_tagger.tag(nltk.word_tokenize(sentence)))


# get common nouns ("NN", "NNS")
def get_common_nouns(sentence):
    """
    Filter and retrieve common-nouns from the sentence (to be candidates for nominaliation)
    :param sentence: unicode string
    :return: list of (index, token) tuples
    """
    COMMON_NOUNS_POS = ["NN", "NNS"]
    nouns = []
    for index, (token, pos) in enumerate(pos_tag(sentence)):
        if pos in COMMON_NOUNS_POS:
            nouns.append((index, token))
    return nouns


# filter by wordnet / catvar / possible-noms
def get_candidate_nouns(sentence,
                        wordnet=True,
                        catvar=True,
                        affixes_heuristic=False):
    """
    Filter and retrieve nom-candidates from the sentence.
    Use "or" on any of 3 filters: WordNet, CatVar, and an artificial list of potential noms
    :param sentence:
    :return:
    """
    nouns = get_common_nouns(sentence)
    candidate_filter = lambda nn: get_verb_forms_from_lexical_resources(nn,
                                                                        wordnet=wordnet,
                                                                        catvar=catvar,
                                                                        affixes_heuristic=affixes_heuristic)[1]
    candidates = [(index, token)
                  for (index, token) in nouns
                  if candidate_filter(token)]
    return candidates


# verbalize using both WordNet and CatVar
def get_verb_forms_from_lexical_resources(nn,
                                          wordnet=True,
                                          catvar=True,
                                          affixes_heuristic=False):
    wordnet_verbs = wordnet_util.convert_pos(nn, wordnet_util.WN_NOUN, wordnet_util.WN_VERB) \
        if wordnet else []
    catvar_verbs = catvar_util.catvariate(nn) if catvar else []
    affixes_heuristic_verbs = verb_to_nom.get_source_verbs(nn) if affixes_heuristic else []
    # sort by distance
    vrbs = wordnet_verbs + catvar_verbs + affixes_heuristic_verbs
    vrbs = [v for v, w in wordnet_util.results_by_edit_distance(nn, vrbs)]
    if vrbs:
        return vrbs, True
    else:
        return [nn], False


# for debug - see candidates
def get_candidate_nouns_from_csv(csv_fn):
    sentences = get_sentences_from_csv(csv_fn)
    all_candidates = [(sid, i, nn)
                      for sid, sentence in sentences.iteritems()
                      for i, nn in get_candidate_nouns(sentence)
                      ]
    return all_candidates


def get_candidate_nouns_from_raw_csv(csv_fn, **resources):
    """
    @:param csv_fn: csv file containing raw sentences
    Returns: list of candidate_info (dict)
    """
    sentences = get_sentences_from_csv(csv_fn)
    all_candidates = []
    for sid, sentence in sentences.iteritems():
        tokenized_sent = nltk.word_tokenize(sentence)
        for idx, nn in get_candidate_nouns(sentence, **resources):
            verb_forms, is_had_verbs = get_verb_forms_from_lexical_resources(nn, **resources)
            candidate_info = {"context": sentence,
                              "intent": idx,
                              "noun": tokenized_sent[idx],
                              "verb_forms": ' '.join(verb_forms[:5]),
                              "sentence_id": sid,
                              "is_factuality_event": "yes",
                              "is_had_auto_verbs": "yes" if is_had_verbs else "no"}
            all_candidates.append(candidate_info)
    return all_candidates

# return prompts DataFrame for identification task - final method for getting
# the well-structured dataframe as input for the simple MTurk task
def get_candidate_nouns_df_from_raw_csv(csv_fn, **resources):
    all_candidates = get_candidate_nouns_from_raw_csv(csv_fn, **resources)
    return pd.DataFrame(all_candidates)


# generate prompts DataFrame for identification task
def generate_nom_id_prompt(raw_sentences_csv_fn, output_prompts_csv_fn):
    prompts_df = get_candidate_nouns_df_from_raw_csv(raw_sentences_csv_fn)
    prompts_df.to_csv(output_prompts_csv_fn)
