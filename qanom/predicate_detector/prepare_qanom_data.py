import csv
from collections import defaultdict
from pathlib import Path
import argparse


def csv2conll(csv_path, conll_path):
    id2sent = dict()
    id2nom = defaultdict(lambda: defaultdict(str))
    with open(csv_path, encoding='utf8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            qasrl_id = row['qasrl_id']
            target_idx = int(row['target_idx'])  # taget ID
            sentence = row['sentence']
            if qasrl_id not in id2sent:
                id2sent[qasrl_id] = sentence
            else:
                assert id2sent[qasrl_id] == sentence
            if 'is_verbal' not in row:
                row['is_verbal'] = 'True'
            is_verbal = row['is_verbal']
            assert is_verbal == 'True' or is_verbal == 'False'

            if target_idx not in id2nom[qasrl_id]:
                id2nom[qasrl_id][target_idx] = is_verbal
            else:
                assert id2nom[qasrl_id][target_idx] == is_verbal

    num_nom_candidates = 0
    with open(conll_path, 'w', encoding='utf8') as f:
        for qasrl_id in id2sent:
            for i, w in enumerate(id2sent[qasrl_id].split()):
                if i in id2nom[qasrl_id]:
                    f.write(w + ' ' + id2nom[qasrl_id][i] + '\n')
                    num_nom_candidates += 1
                else:
                    f.write(w + ' ' + 'O' + '\n')
            f.write('\n')
    print(csv_path, "# Nom candidates: ", num_nom_candidates)


parser = argparse.ArgumentParser(description='Prepare QANOM data: format data to generate files '
                                             'in CoNLL format given the CSV files produced '
                                             'during candidate extraction.')
parser.add_argument('--INPUT_DIR', type=str, default='output/candidate_extraction/',
                    help="directory with CSV formatted candidate extraction output ("
                         "train/dev/test split).")
parser.add_argument('--OUTPUT_DIR', type=str, default='output/predicate_detector/',
                    help="directory with CoNLL formatted candidate extraction output ("
                         "train/dev/test split).")
parser.add_argument('--INPUT_FILE', type=str, help="CSV formatted candidate extraction output.")
parser.add_argument('--OUTPUT_FILE', type=str, help="CoNLL formatted candidate extraction output.")
args = parser.parse_args()

inputs = []
outputs = []

if args.INPUT_FILE is not None:
    inputs.append(Path(args.INPUT_FILE))
    outputs.append(Path(args.OUTPUT_FILE))
else:
    if args.INPUT_DIR is not None:
        for split in {'dev', 'test', 'train'}:
            inputs.append(Path(args.INPUT_DIR, split + '.csv'))
            outputs.append(Path(args.OUTPUT_DIR, split + '.txt'))

for (csv_path, conll_path) in zip(inputs, outputs):
    csv2conll(csv_path, conll_path)
