import sys, os, catvar, pattern
from nltk import wordnet as wn
from nltk.corpus import verbnet
# lists of verbs
# from pattern - 8.5K
all_pattern_verbs = pattern.en.verbs.infinitives.keys()
# from wordnet - 8.7K
all_wn_verbs = sorted(set(l.name()
                for v_syn in wn.wordnet.all_synsets(pos="v")
                for l in v_syn.lemmas()
                if "_" not in l.name()))
# from verbnet - 3.6K
all_verbnet_verbs = verbnet.lemmas()
# together - 10.6K
infinitives = sorted(set(all_wn_verbs + all_pattern_verbs + all_verbnet_verbs))

def as_gerund(verb):
    return pattern.en.conjugate(verb, aspect=pattern.en.PROGRESSIVE)

def with_suffix(verb, suffix, replace=[]):
    for suf_to_replace in sorted(replace, key=lambda s:len(s), reverse=True):
        if verb.endswith(suf_to_replace):
            return verb.rstrip(suf_to_replace) + suffix
    return verb + suffix

def replace_suffix(verb, replacements):
    # replacements is dict {"current-suffix" : "new-suffix"}
    for suf, new_suf in sorted(replacements.items(), key=lambda (k,v): len(k), reverse=True):
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


def get_deverbal_pairs(verb_seed=infinitives):
    # verb_seed is a list of verbs (infinitive form)
    return {nom : verb
            for verb in verb_seed
            for nom in verb_to_all_possible_nominalizations(verb)}


def get_all_possible_nominalizations(verb_seed=infinitives):
    return get_deverbal_pairs(verb_seed).keys()
