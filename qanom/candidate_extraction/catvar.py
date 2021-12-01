from qanom import config
import os

catvar_corpus_relative_path = "catvar/catvar21.signed" # should be in 'resources' directory
catvar_corpus_path = os.path.join(config.resources_path, catvar_corpus_relative_path)

def get_catvar_pairs():
    # catvar_pairs is a list of 16K tuples (nominalization, source-verb) for 13.8K nominalizations

    catvar_clusters = []
    with open(catvar_corpus_path, "r") as f:
        for line in f.readlines():
            cluster = [] # cluster of Words, each Word is a tuple (word-surface, pos, signature-num)
            for word_info in line.strip().split("#"):
                word_and_pos, signature = word_info.split("%")
                word, pos = word_and_pos.split("_")
                cluster.append((word, pos, int(signature)))
            catvar_clusters.append(cluster)
    def get_pairs(c):
        for noun in [w[0] for w in c if w[1]=='N']:
            for verb in [w[0] for w in c if w[1]=='V']:
                yield (noun, verb)
    catvar_pairs = [list(get_pairs(c))
                   for c in catvar_clusters
                   if len(c) > 1]
    flatten = lambda l: [item for sublist in l for item in sublist]
    catvar_pairs = flatten(catvar_pairs)
    return catvar_pairs
catvar_pairs = get_catvar_pairs()

# catvar_pairs contains 16810 pairs for 13788 nouns (keys)
def catvariate(noun):
    return [verb for n,verb in catvar_pairs if n==noun]