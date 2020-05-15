# generate Data CSVs for QASRL-2.0 data jsonl files
import csv
import json

import pandas as pd

HEADER = ["qasrl_id", "domain", "tokens", "sentence"]


def getDomain(id):
    if id.split(":")[0] == "TQA":
        return "TQA"
    else:
        return id.split(":")[1]


def get_qasrl_corpus_data(fn):
    ids = []
    corpusData = []
    with open(fn, "r", encoding='latin-1') as f:
        for line in f:
            sentData = json.loads(line)
            sentId = sentData["sentenceId"]
            sentTokens = sentData["sentenceTokens"]
            domain = getDomain(sentId)
            sent = ' '.join(sentTokens)
            ids.append(sentId)
            corpusData.append([sentId, domain, sent, sent])
    return pd.DataFrame(corpusData, columns=HEADER)


def writeToCSV(csv_fn, list_of_lineLists, header=None):
    with open(csv_fn, "w") as outcsv:
        writer = csv.writer(outcsv)
        if header:
            writer.writerow(header)
        writer.writerows(list_of_lineLists)

#corpusData=get_qasrl_corpus_data(sys.argv[1])
#writeToCSV(fn + ".csv", corpusData, HEADER)

from typing import *

def split_long_sentence_df(ddf: Dict[str, pd.DataFrame], dest_dir: str, size_of_each: int = 1000):
    """
    :param ddf: {domain-name: dataFrame of sentences of that domain}
    :return:
    """
    import math
    for domain in ["wikinews", "wikipedia"]:
        long_df = ddf[domain].reset_index(drop=True)
        n_files = math.ceil(long_df.shape[0] / size_of_each)
        for i in range(1,n_files+1):
            df = long_df.loc[(i-1)*size_of_each : i*size_of_each]
            df.to_csv(f"{dest_dir}/{domain}.train.{i}.csv", index=False)
