"""
Use lexical resources to extract nouns that are candidates for being deverbal nominalizations.

The script can be used for generating a JSON file that serves as input to the qasrl-crowdsourcing MTurk application
for annotating verbal arguments of nominalizations (qanom-unified branch).
It can also be used for generating a CSV files that serves as input to the predicate_detector.


Logic building blocks:
* load raw data  (sentences) from corpus
* use Stanford CoreNLP to POS-tag the sentence
* get common nouns ("NN", "NNS")
* filter by wordnet + CatVar (+ possible-noms from the inhouse verb_to_nom utility)
* generate a list of nominalization candidates, each containing:
    { "sentenceId": sentence ID (string),
      "tokSent": sentence tokens (list),
      "targetIdx": index of candidate noun (int),
      "verbForms": non-empty list of verb-forms (strings) (ordered by edit distance)
    }

"""

import json, os, sys
from typing import List, Tuple, Iterable, Dict, Any

import pandas as pd
import nltk
from nltk.downloader import Downloader

# add project basic directory to sys.path, in order to refer qanom as a package from anywhere
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)

from qanom.candidate_extraction import cand_utils
from qanom.annotations.common import read_csv
from qanom.candidate_extraction.verb_to_nom import SuffixBasedNominalizationCandidates as VTN

""" Define which resources should be used (by default) for filtering nouns as candidate nominalizations. """
default_resources = {"wordnet": True,
                     "catvar": True,
                     "affixes_heuristic": True}

vtn = None # init this global VTN object only if required

# by default, use nltk's default pos_tagger ('averaged_perceptron_tagger'):
tagger_package = 'averaged_perceptron_tagger'
nltk_downloader = Downloader()
if (not nltk_downloader.is_installed(tagger_package)):
    nltk.download(tagger_package)
    
pos_tag = nltk.pos_tag
"""
Alternatively, when extracting candidates for crowdsourcing QANom annotations through the qasrl-crowdsourcing project,
one should use the same POS model as inside qasrl-crowdsourcing for consistency. 
qasrl-crowdsourcing uses the CoreNLPParser model in Java. We will use here nltk's CoreNLPParser wrapper.
To run the CoreNLPParser model as a server on your machine (port 9000), 
pre-run the following command from the unzipped directory of the stanford-core-nlp project 
(see https://www.khalidalnajjar.com/setup-use-stanford-corenlp-server-python/ for instructions): 
```bash
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000
```
Then, replace the above two-lines in the python script with the following block: 
```python
from nltk.parse import CoreNLPParser
pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
pos_tag = pos_tagger.tag
```
"""

COMMON_NOUNS_POS = ["NN", "NNS"]

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

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

    wordnet_verbs = []
    if wordnet:
        from qanom.candidate_extraction import wordnet_util
        wordnet_verbs = wordnet_util.convert_pos_by_lemmas(nn, wordnet_util.WN_NOUN, wordnet_util.WN_VERB)
        # apply distant-verbs filtering on wordnet verbs
        if filter_distant:
            wordnet_verbs = list(filter(lambda v: filter_distant_verb_forms(v, nn), wordnet_verbs))

    catvar_verbs = []
    if catvar:
        from qanom.candidate_extraction import catvar as catvar_util
        catvar_verbs = catvar_util.catvariate(nn)

    affixes_heuristic_verbs = []
    if affixes_heuristic:
        global vtn
        if vtn is None: # initialize global vtn only once, if required
            vtn = VTN()
        affixes_heuristic_verbs = vtn.get_source_verbs(nn)
    # sort by distance
    vrbs = catvar_verbs + wordnet_verbs + affixes_heuristic_verbs
    vrbs = [v for v, w in cand_utils.results_by_edit_distance(nn, vrbs)]
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
    edit_distance = cand_utils.levenshteinDistance(verb_form, noun)
    short_size, long_size = min(len(verb_form),len(noun)), max(len(verb_form),len(noun))
    edit_distance_without_suffix = cand_utils.levenshteinDistance(verb_form[:short_size], noun[:short_size])
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
def get_sentences_from_allennlp_jsonl(sentences_josnl_fn: str) -> Dict[str, str]:
    """
    @:param sentences_josnl_fn: a file name of JSON-lines of input sentences, as allennlp predictors input format:
        {"sentence": <actual sentence string> }
        {"sentence": <actual sentence string> }
        ...
    Returns: dict of sentences. Generated a unique sentence_id for the sentences (by index).
    """
    with open(sentences_josnl_fn) as f:
        sentences = []
        for line in f.readlines():
            sentences.append(json.loads(line)["sentence"])
    return get_sentences_from_iterable(sentences)


def get_sentences_from_iterable(sentences: Iterable[str]) -> Dict[str, str]:
    """
    @:param sentences: iterable of raw sentences
    Returns: dict of sentences. Generated a unique sentence_id for the sentences (by index).
    """
    sentences_dict: Dict[str, str] = {f"Sentence-{1+i:04}": sent for i, sent in enumerate(sentences)}
    return sentences_dict


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
    df['sentence'] = df.apply(lambda r: ' '.join(r['tokSent']), axis=1)
    df['noun'] = df.apply(lambda r: r['tokSent'][int(r['targetIdx'])], axis=1)
    df['verb_form'] = df.apply(lambda r: r['verbForms'][0], axis=1)
    df = df.drop(['tokSent', 'verbForms'], axis='columns')
    df = df.rename(mapper={'targetIdx': 'target_idx',
                           'sentenceId': 'qasrl_id'},
                   axis='columns')
    if csv_out_fn:
        df.to_csv(csv_out_fn, index=False)
    return df

# a Utility function to use candidate extraction both as a module as well as script
def extract_candidate_nouns(input: Any,
                            input_format: str,  # Literal["iterable", "raw", "dict", "csv", "jsonl"]
                            **resources) -> List[Dict[str, Any]]:
    # get sentences-info into a {sentence_id: sentence} dict
    if input_format == "dict":
        sentences = input
    elif input_format == "iterable":
        sentences = get_sentences_from_iterable(input)
    elif input_format == 'raw':
        with open(input) as fin:
            sentences = get_sentences_from_iterable(fin.read().splitlines())
    elif input_format == 'csv':
        sentences = get_sentences_from_csv(input)
    elif input_format == 'jsonl':
        sentences = get_sentences_from_allennlp_jsonl(input)
    else:
        raise NotImplementedError('input_format must be "iterable", "dict", "csv" or "jsonl".')

    candidates = get_candidate_nouns(sentences, **resources)
    return candidates


def main(args):
    """ Expecting command line arguments from `argparse`. """

    # determine resources
    resources = {"wordnet": args.wordnet,
                 "catvar": args.catvar,
                 "affixes_heuristic": args.affixes_heuristic}

    candidates = extract_candidate_nouns(args.sentences_fn,
                                         args.input_format,
                                         **resources)

    # Write candidates information to output file. """
    if args.output_format == 'json':
        json.dump(candidates, open(args.output_fn, "w"))
    elif args.output_format == 'csv':
        export_candidate_info_to_csv(candidates, args.output_fn)

