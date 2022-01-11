"""
Use lexical resources to extract nouns that are candidates for being deverbal nominalizations.

The script can be used for generating a JSON file that serves as input to the qasrl-crowdsourcing MTurk application
for annotating verbal arguments of nominalizations (qanom-unified branch).
It can also be used for generating a CSV files that serves as input to the predicate_detector.
"""

import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Use lexical resources to extract nouns that are " +
                                                 "candidates for being deverbal nominalizations.")
    ap.add_argument('sentences_fn', type=str)
    ap.add_argument('output_fn', type=str)
    ap.add_argument('--read', dest='input_format', choices=['csv', 'jsonl', 'raw'], default='csv',
                    help="Define the format of sentences_fn to read from. \n"
                        +"csv is expecting a 'sentence' column and a 'sentence_id|sentenceId|qasrl_id' column.\n"
                        +"jsonl correspond to AllenNLP predictor's format, where each line is {'sentence': string}.\n"
                        +"raw is a text file, where each sentence is in a new line.")

    ap.add_argument('--write', dest='output_format', choices=['csv', 'json'], default='json',
                    help="Define the output format of candidate information. \n"
                             + "csv is the QANom default format. This is the format which predicate-detector model expects as input. \n"
                             + "json is used as input in the qasrl-crowdsourcing system when crowdsourcing QANom annotations.")
    # which resources to use - by default, use all three
    ap.add_argument('--no-wordnet', dest='wordnet', action='store_false')
    ap.add_argument('--no-catvar', dest='catvar', action='store_false')
    ap.add_argument('--no-affixes', dest='affixes_heuristic', action='store_false')

    args = ap.parse_args()

    from qanom.candidate_extraction import candidate_extraction
    candidate_extraction.main(args)
