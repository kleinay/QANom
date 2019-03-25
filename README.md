# QANom - Annotating Nominal Predicates with QA-SRL

## Retrieving cadidate nouns as input for the Nominalization-Identification task

The entry point is the script `prepare_nom_ident_batch.py`.
There, the function used to filter candidate nouns from a sentence string is `get_candidate_nouns`. If you'll have all the requirements you can use it as well, otherwise you should modify it a bit or modify the functions it is calling. It uses a CoreNLP running server to `pos_tag` the sentence, and filter outs anything except common nouns (`get_common_nouns`). Then, it applies another filter - an "or" combination of two kinds of filtering algorithms:
1. Lexical Resources based - Use WordNet & CatVar derivations. Any noun with verbal related derivation would be predicted as nom.
2. Affixes + seed based - Create a (possible-nominalization -> verb) list out of a verb seed list, using simple nom. affixes substitution rules. Being in the list will be considered as being a nominalization candidate.

Filter 1 (`wordnet_util.py` + `catvar.py`) requires wordnet (available via [nltk](https://www.nltk.org/)) and [CatVar](https://clipdemos.umiacs.umd.edu/catvar/) (contact to get it; extract into "resources" dir under main QANom dir. Doesn't require installation).
Filter 2 (`verb_to_nom.py`) uses [pattern.en](https://www.clips.uantwerpen.be/pages/pattern-en) package.  

