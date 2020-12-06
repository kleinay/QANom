import argparse

from qanom.annotations.common import read_csv


def conll2csv(conll_path, input_csv_path, output_csv_path):
    with open(conll_path, 'r', encoding='utf8') as f:
        sentence = []
        tags = []
        sent2tag = dict()
        for line in f:
            line = line.strip()
            if len(line) > 0:
                sentence.append(line.split()[0])
                tags.append(line.split()[1])
            else:
                if len(sentence) > 0:
                    sent2tag[' '.join(sentence)] = tags
                    sentence = []
                    tags = []

    if len(sentence) > 0:
        sent2tag[' '.join(sentence)] = tags

    df = read_csv(input_csv_path)
    for i, row in df.iterrows():
        df.at[i, 'is_verbal'] = sent2tag[row['sentence']][row['target_idx']]

    df.to_csv(output_csv_path)


parser = argparse.ArgumentParser(description='Convert CoNLL file produced by predicate detector '
                                             'to CSV format given the CSV file produced '
                                             'during candidate extraction.')
parser.add_argument('--INPUT_CONLL_FILE', type=str, help="CoNLL formatted predicate detector "
                                                         "output.",
                    default='output/predicate_detector/test_predictions.txt')
parser.add_argument('--INPUT_CSV_FILE', type=str, help="CSV formatted candidate extraction "
                                                       "output.",
                    default='output/candidate_extraction/test.csv')
parser.add_argument('--OUTPUT_FILE', type=str, help="CSV formatted predicate detector output.",
                    default='output/predicate_detector/test_predictions.csv')
args = parser.parse_args()

conll2csv(args.INPUT_CONLL_FILE, args.INPUT_CSV_FILE, args.OUTPUT_FILE)
