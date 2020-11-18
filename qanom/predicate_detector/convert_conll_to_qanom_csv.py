import csv
import argparse


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

    with open(input_csv_path, encoding='utf8') as csv_file, open(output_csv_path, 'w',
                                                                 encoding='utf8') as csv_pred_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_writer = csv.writer(csv_pred_file,delimiter=',')
        header = next(csv_reader)  # skip header
        csv_writer.writerow(header)
        for row in csv_reader:
            target_idx = row[2]
            sentence = row[1]
            row[6] = sent2tag[sentence][int(target_idx)]
            csv_writer.writerow(row)


parser = argparse.ArgumentParser(description='Convert CoNLL file produced by predicate detector '
                                             'to CSV format given the CSV file produced '
                                             'during candidate extraction.')
parser.add_argument('--INPUT_CONLL_FILE', type=str, help="CoNLL formatted predicate detector "
                                                         "output.",
                    default='output/predicate_detector/test_predictions.txt')
parser.add_argument('--INPUT_CSV_FILE', type=str, help="CSV formatted candidate extraction "
                                                       "output.", default='dataset/test.csv')
parser.add_argument('--OUTPUT_FILE', type=str, help="CSV formatted predicate detector output.",
                    default='output/predicate_detector/test_predictions.csv')
args = parser.parse_args()

conll2csv(args.INPUT_CONLL_FILE, args.INPUT_CSV_FILE, args.OUTPUT_FILE)