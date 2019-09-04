# generate Data CSVs for QASRL-2.0 data jsonl files
import sys, json, csv, pandas as pd
HEADER = ["sentenceId", "domain", "tokens", "sentence"]


def getDomain(id):
    if id.split(":")[0] == "TQA":
        return "TQA"
    else:
        return id.split(":")[1]


def get_qasrl_corpus_data(fn):
    ids = []
    corpusData = []
    with open(fn, "r") as f:
        for line in f:
            sentData = json.loads(line)
            sentId = sentData["sentenceId"]
            sentTokens = sentData["sentenceTokens"]
            domain = getDomain(sentId)
            sent = u' '.join(sentTokens).encode('utf-8')
            ids.append(sentId)
            corpusData.append([sentId, domain, sent, sent])
    return pd.DataFrame(corpusData, header=HEADER)


def writeToCSV(csv_fn, list_of_lineLists, header=None):
    with open(csv_fn, "w") as outcsv:
        writer = csv.writer(outcsv)
        if header:
            writer.writerow(header)
        writer.writerows(list_of_lineLists)

#corpusData=get_qasrl_corpus_data(sys.argv[1])
#writeToCSV(fn + ".csv", corpusData, HEADER)


