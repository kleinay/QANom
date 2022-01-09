import csv
from collections import defaultdict
import argparse

def csv2conll(csv_filename, conll_filename):
    id2sent = dict()
    id2nom = defaultdict(lambda: defaultdict(str))
    with open(csv_filename, encoding='utf8') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        header_row = next(csv_reader)  # skip header
        #idx = header_row.index(column)
        for row in csv_reader:
            qasrl_id = row['qasrl_id']
            target_idx = int(row['target_idx']) # taget ID
            sentence = row['sentence']
            if qasrl_id not in id2sent:
                id2sent[qasrl_id] = sentence
            else:
                assert id2sent[qasrl_id] == sentence
            is_verbal = row['is_verbal']
            assert is_verbal == 'True' or is_verbal == 'False'

            if target_idx not in id2nom[qasrl_id]:
                id2nom[qasrl_id][target_idx] = is_verbal
            else:
                assert id2nom[qasrl_id][target_idx] == is_verbal

    num_nom_candidates = 0
    with open(conll_filename, 'w', encoding='utf8') as f:
        for qasrl_id in id2sent:
            for i, w in enumerate(id2sent[qasrl_id].split()):
                if i in id2nom[qasrl_id]:
                    f.write(w + ' ' + id2nom[qasrl_id][i] + '\n')
                    num_nom_candidates += 1
                else:
                    f.write(w + ' ' + 'O' + '\n')
            f.write('\n')
    print(csv_filename, "# Nom candidates: ", num_nom_candidates)


parser = argparse.ArgumentParser(description='Prepare QANOM data')
parser.add_argument('--INPUT_DIR', type=str, default='output/candidate_extraction/',
                    help="directory with candidate extraction output")
parser.add_argument('--OUTPUT_DIR', type=str, default='output/predicate_detector/',
                    help="directory with predicate detector output")
args = parser.parse_args()

for mode in {'dev', 'test', 'train'}:
    csv2conll(f"{args.INPUT_DIR}/annot.{mode}.csv", f"{args.OUTPUT_DIR}/{mode}.txt")