import pattern3.en as pattern
from nltk import wordnet as wn
from nltk.corpus import verbnet
# lists of verbs
# from pattern - 8.5K
all_pattern_verbs = list(pattern.verbs.infinitives.keys())
# from wordnet - 8.7K
all_wn_verbs = sorted(set(l.name()
                for v_syn in wn.wordnet.all_synsets(pos="v")
                for l in v_syn.lemmas()
                if "_" not in l.name()))
# from verbnet - 3.6K
all_verbnet_verbs = list(verbnet.lemmas())
# together - 10.6K
infinitives = sorted(set(all_wn_verbs + all_pattern_verbs + all_verbnet_verbs))

def as_gerund(verb):
    return pattern.conjugate(verb, aspect=pattern.PROGRESSIVE)

def with_suffix(verb, suffix, replace=[]):
    for suf_to_replace in sorted(replace, key=lambda s:len(s), reverse=True):
        if verb.endswith(suf_to_replace):
            return verb.rstrip(suf_to_replace) + suffix
    return verb + suffix

def replace_suffix(verb, replacements):
    # replacements is dict {"current-suffix" : "new-suffix"}
    for suf, new_suf in sorted(replacements.items(), key=lambda pair: len(pair[0]), reverse=True):
        if verb.endswith(suf):
            return verb.rstrip(suf) + new_suf
    return verb

def verb_to_all_possible_nominalizations(verb):
    # returns ~17 possible nominal forms of the verb, using morphology only (some are probably not words)
    noms = [verb,
            str(as_gerund(verb)),
            # -tion suffix: ("construction")
            with_suffix(verb, suffix="tion", replace=["t", "te"]),
            replace_suffix(verb, {"ire": "iration",
                                  "ume": "umption",
                                  "quire": "quisition",
                                  "lve": "lution"}),
            # -sion suffix: ("conclusion", "transmission")
            replace_suffix(verb, {"it": "ission",
                                  "t": "sion",
                                  "de": "sion",
                                  "d": "sion",
                                  "se": "sion",
                                  "ss": "ssion"
                                  }),
            # -ment suffix: ("development")
            with_suffix(verb, suffix="ment"),
            # -ance suffix: ("acceptance")
            replace_suffix(verb, {"ance": "ance",
                                  "e": "ance",
                                  "": "ance",
                                  "y": "iance"}),
            # -ence suffix: ("reference")
            replace_suffix(verb, {"ent": "ence",
                                  "e": "ence",
                                  "": "ence"}),
            # -ure suffix: ("failure")
            with_suffix(verb, suffix="ure", replace=["e", "ure"]),
            # -al suffix: (e.g. proposal, withdrawal)
            with_suffix(verb, suffix="al", replace=["e", "al"]),
           ]
    plural_nominalizations = map(pattern.en.pluralize, set(noms))
    all_verb_derived_noms = list(set(noms + plural_nominalizations))
    return all_verb_derived_noms


def get_deverbal_dict(verb_seed=infinitives, multipleEntriesPerNom=False):
    """
    verb_seed is a list of verbs (infinitive form).
    Returns:
        multipleEntriesPerNom==False:   a dict { nom : source-verb }
        multipleEntriesPerNom==True:    a dict { nom : [source-verbs] }
    Note: when multipleEntriesPerNom == False, function is not handling cases where
    the same affix-based potential nominalization is derived from multiple verbs.
    The dict will conflate these arbitrarily.
    """
    pairs = get_deverbal_pairs(verb_seed=verb_seed)
    if multipleEntriesPerNom:
        import scripts.utils
        return scripts.utils.dictOfLists(pairs)
    else:
        return dict(pairs)

def get_deverbal_pairs(verb_seed=infinitives):
    """
    verb_seed is a list of verbs (infinitive form).
    Return a list of (possible-nom, source-verb) based on affix heuristics for each verb in verb_seed.
    """
    return [(nom, verb)
            for verb in verb_seed
            for nom in verb_to_all_possible_nominalizations(verb)]


def get_all_possible_nominalizations(verb_seed=infinitives):
    return get_deverbal_dict(verb_seed).keys()

# End usage - create verb_to_nom list and query noun for source verbs
def get_source_verbs(nn):
    """ Return a list of source verbs per noun if noun is in artificially generated verb_to_nom list.
        If noun is not in list, returns empty list.
    """
    affixes_based_nom_dict = get_deverbal_dict(multipleEntriesPerNom=True)
    if nn in affixes_based_nom_dict:
        return affixes_based_nom_dict[nn]
    else:
        return []
