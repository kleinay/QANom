# (C) Gabriel Stanovsky
from nltk.corpus import wordnet as wn
from qanom.candidate_extraction import cand_utils
# Just to make it a bit more readable
WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'


def convert_pos_by_lemmas(word, from_pos, to_pos):
    """ Transform words given from/to POS tags """

    lemmas = wn.lemmas(word, pos=from_pos)
    # Word not found
    if not lemmas:
        return []

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = []

    for drf in derivationally_related_forms:
        for l in drf[1]:
            if l.synset().name().split('.')[1] == to_pos or \
                (to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
                 and l.synset().name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)):
                related_noun_lemmas += [l]

    # Extract the words from the lemmas
    related_words = [l.name() for l in related_noun_lemmas]
    return list(set(related_words))


def convert_pos_by_synset(word, from_pos, to_pos):
    """ Transform words given from/to POS tags """

    synsets = wn.synsets(word, pos=from_pos)
    # Word not found
    if not synsets:
        return []

    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = []
    for s in synsets:
        for l in s.lemmas():
            if s.name().split('.')[1] == from_pos or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and s.name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                lemmas += [l]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = []

    for drf in derivationally_related_forms:
        for l in drf[1]:
            if l.synset().name().split('.')[1] == to_pos or \
                (to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
                 and l.synset().name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)):
                related_noun_lemmas += [l]

    # Extract the words from the lemmas
    related_words = [l.name() for l in related_noun_lemmas]
    return list(set(related_words))




def verbalize(word, from_pos=WN_NOUN):
    related_verbs = convert_pos_by_lemmas(word, from_pos, WN_VERB)
    return cand_utils.results_by_edit_distance(word, related_verbs)


if __name__ == "__main__":
    # example usage
    print(convert_pos_by_lemmas("recognition", WN_NOUN, WN_VERB))
