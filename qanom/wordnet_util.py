# (C) Gabriel Stanovsky
from nltk.corpus import wordnet as wn

# Just to make it a bit more readable
WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'

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


def results_by_edit_distance(orig_word, optional_results):
    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, levenshteinDistance(w, orig_word)) for w in set(optional_results)]
    result.sort(key=lambda w:len(w[0])) # tie breaker - prefer shorter verbs
    result.sort(key=lambda w:-w[1], reverse = True)

    # return all the possibilities sorted by probability
    return result


def verbalize(word, from_pos=WN_NOUN):
    related_verbs = convert_pos_by_lemmas(word, from_pos, WN_VERB)
    return results_by_edit_distance(word, related_verbs)


if __name__ == "__main__":
    # example usage
    print(convert_pos_by_lemmas("recognition", WN_NOUN, WN_VERB))
