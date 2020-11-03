""" Utils for candidate_extraction module """

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


def results_by_edit_distance(orig_word, optional_results):
    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, levenshteinDistance(w, orig_word)) for w in set(optional_results)]
    result.sort(key=lambda w:len(w[0])) # tie breaker - prefer shorter verbs
    result.sort(key=lambda w:-w[1], reverse = True)

    # return all the possibilities sorted by probability
    return result
