from argparse import ArgumentParser
import sys, os

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("system_path", help="system predicted annotations (CSV file)")
    ap.add_argument("ground_truth_path", help="reference annotations (CSV file")
    ap.add_argument("sentences_path", default=None,
                    help="if the input files don't have a `sentence` column with the raw sentences (but only `qasrl_id`), provide a CSV file that maps `qasrl_id` to `sentence`.")
    args = ap.parse_args()

    from qanom.evaluation import evaluate
    evaluate.main(args.system_path, args.ground_truth_path, args.sentences_path)
