# QANom - Annotating Nominal Predicates with QA-SRL - paper full reference 


QANom is a research project aiming for a natural representation of nominalization's predicate-argument relations.
It extends the Question Answer driven Semantic Role Labeling (QASRL) framework (see [website](http://qasrl.org/)), which tackled verbal predicates,
to the more challenging space of deverbal nominalizations. 

This repository is the reference point for the data and software described in the paper 
[*QANom: Question-Answer driven SRL for Nominalizations*](https://www.aclweb.org/anthology/2020.coling-main.274/) (COLING 2020).



## Pre-requisite
* Python 3.7

## Installation
From pypi:
`pip install qanom`

If you want to install from source, clone this repository and then install requirements:
```bash
git clone https://github.com/kleinay/QANom.git
cd QANom
pip install requirements.txt
```



## Dataset

The original QANom Dataset can be downloaded from this
 [google drive directory](https://drive.google.com/drive/folders/15PHKVdPm65ysgdkV47z6J_73kETk7_of).
 
 Alternatively, if you are working with Huggingface's [Datasets](https://github.com/huggingface/datasets) library or are willing to install it (`pip install datasets`), 
 you can retrieve the QANom datasets by:
 ```python
 import datasets
 qanom_dataset = datasets.load_dataset('biu-nlp/qanom')
 ```

## Crowdsourcing QANom via MTurk

The QANom dataset was collected through [Amazon Mechanical Turk](https://www.mturk.com/) (MTurk). 
It was annotated by pre-selected crowd workers who exhibited good performance when previously annotating QA-SRL.
Workers first thoroughly read and comprehend the **annotation guidelines** - 
both for [question generation](https://docs.google.com/presentation/d/1AGLdjilE4GDaF1ybXaS4JXabGLrfK58W1p6mteU_yrw/present?slide=id.p) 
and for [QA consolidation](https://docs.google.com/presentation/d/1ECharO3EKCabVDx_PYVUDfbdgYJ0uKu5sRo165OSNXM/present?slide=id.p).
   
We adjusted (on a side branch) the [qasrl-crowdsourcing](https://github.com/kleinay/qasrl-crowdsourcing/tree/qanom_unified_oneverb) software 
in order to run the QANom Annotation task interface on MTurk. 
The QANom annotation pipeline is different from QA-SRL pipeline in three aspects:
1. QANom annotation is running after raw sentences have been preprocessed with the `candidate_extraction` module, 
which results in a JSON file describing the nominalization candidates and their heuristically extracted verbal counterparts (`verb_form`). 
In our branch, the `qasrl-crowdsourcing` system is expecting this JSON input instead of raw text input. 
2. The annotation tasks are running on MTurk, but only workers granted with our *Qualifications* can work on the task.
There are unique qualifications for the different annotation life-cycle phases (Trap, Training, Production) 
as well of for the different tasks (Generation, Consolidation).   
3. In the QANom task interface, the QA-SRL annotation phase 
(i.e. generating QA-SRL questions and highlighting their answers in the sentence) 
is preceded with a predicate detection yes/no question.

**Note**: the original [qasrl-crowdsourcing](https://github.com/julianmichael/qasrl-crowdsourcing) 
package has been refactored since our usage, which might result in some installation errors. 
Feel free to contact us for guidance.  


## Candidate Extraction
You can use the candidate extraction module to for several purposes:
1. As a pre-processing for the QANom crowd-annotation task
2. As a pre-processing for the `predicate_detector` model
3. For custom usage.

### Module-specific pre-requisite 
```bash
pip install pandas nltk git+git://github.com/pattern3/pattern
```

### Usage
```bash
python qanom/extract_candidates.py <input_file> <output_file> --read csv|josnl|raw --write csv|json [--no-wordnet] [--no-catvar] [--no-affixes]
```
The script handle three input formats:
* `csv` (default): a comma-separated file, with a `sentence` column stating the raw string of the sentence, 
and a `sentence_id` or `qasrl_id` column for sentence identifiers.   
* `jsonl`: a JSON-lines file, similar to AllenNLP predictors' inputs- each line is `{"sentence": <actual sentence string> }`. 
* `raw`: text file, each sentence is in a new line.

The script handle two output formats:
* `json` (default): used as input in the `qasrl-crowdsourcing` system when crowdsourcing QANom annotations.
* `csv`: QANom Dataset common format. This is the format which the `predicate_detector` model expects as input.

By default, the module uses (the union of) all three filters - `wordnet`, `catvar`, and `affixes_heuristic` (see specification below). 
One can deactivate a filter using the `[--no-wordnet] [--no-catvar] [--no-affixes]` boolean flags.


**Implementation Details**: 

The entry point is the module file `qanom\candidate_extraction\candidate_extraction.py`. 

The module uses a POS tagger to `pos_tag` the sentence, and filter outs anything except common nouns (`get_common_nouns`). 
Then, it applies another filter - an "or" combination of two kinds of lexical-based filtering algorithms:
1. Lexical Resources based - Use WordNet & CatVar derivations. Any noun with verbal related derivation would be predicted as nom.
2. Affixes + seed based - Create a (possible-nominalization -> verb) list out of a verb seed list, using simple nominalization-suffixes substitution rules. 
Being in the list will be considered as being a nominalization candidate.

Filter 1 (`wordnet_util.py` + `catvar.py`) requires 
wordnet (available via [nltk](https://www.nltk.org/)) and [CatVar](https://clipdemos.umiacs.umd.edu/catvar/).
Run `./scripts/download_catvar.sh` for downloading CatVar into the `resources` directory.

Filter 2 (`verb_to_nom.py`) uses [pattern.en](https://www.clips.uantwerpen.be/pages/pattern-en) package (`pip install git+git://github.com/pattern3/pattern`).
 The package is not maintained by the authors and contain a few minor errors that is easy to fix on your local installation.
 The version we are installing here works only for Windows. on Linux, use `--no-affixes` to disable this filter. 

If there are multiple derivationally related verbs, we select the verb that minimizes the edit distance with the noun. 

# Models

The instructions in this section assume you have cloned the QANom repo, and your working directory is the QANom directory. 

## QANom Predicate Detector
The `predicate_detector` classifies nominalization candidates (extracted with the `candidate_extraction` module) as verbal vs. non-verbal. 
We supply a [model](https://drive.google.com/file/d/1qiyQCL19ktZETbPWk_5TT2oCFSFYk6QJ/view?usp=sharing) based on a vanilla BERT-based model 
implemented by fine-tuning bert-base-cased pre-trained model on QANom dataset.

1. Format data to generate files in CoNLL format given the CSV files produced during candidate
 extraction.
```bash
python qanom/predicate_detector/prepare_qanom_data.py [--INPUT_DIR input_dir] [--OUTPUT_DIR
 output_dir]
```

2. If you want to train a new model (else, you can skip to the next step and use the pretrained model):
```bash
sh qanom/predicate_detector/train_nom_id.sh
```

3. Predict using a trained model:
```bash
sh qanom/predicate_detector/predict_nom_id.sh
```

4. Convert CoNLL file produced by predicate detector to QANom's CSV format given the CSV input file:
 produced during candidate extraction.
```bash
python qanom/predicate_detector/convert_conll_to_qanom_csv.py [--INPUT_CONLL_FILE input_conll_file]
                                     [--INPUT_CSV_FILE input_csv_file]
                                     [--OUTPUT_FILE output_file]
```

## QANom Baseline parser 
The `qanom_parser` is essentially the [nrl-qasrl](https://github.com/kleinay/nrl-qasrl/tree/qanom) parser for QA-SRL, presented in 
[*Large-Scale QA-SRL Parsing* (FitzGerald et. al., 2018)](https://www.aclweb.org/anthology/P18-1191/).
To adapt the parser to QANom specifications (e.g. that the verb in the question is not the predicate itself) 
and format (csv), we have our own `qanom` branch on the `nrl-qasrl` repository. This branch uses the `qanom` package.
Run `./scripts/setup_parser.sh` to clone the parser into `qanom_parser` directory and prepare its prerequisites.
Then `cd qanom_parser` to run model-related commands as those described for the rest of this section.

### Training models
Follow the `README` in `qanom_parser` for instructions on training new verbal QA-SRL models.

A QANom parser is trained using a CSV file (QANom format) as input, with the `QANomReader` DatasetReader 
(in `nrl/data/dataset_readers/qanom_reader.py`). 
You should specify the path of the input files in the `jsonnet` config files.
For example: 

```bash
# first train a span-predictor for identifying answer spans (i.e. arguments)
allennlp train configs/train_qanom_span_elmo.jsonnet --include-package nrl -s ../models/<span-model-name> 

# then train the question-generator model, predicting QA-SRL question slots given an answer-span
allennlp train configs/train_qanom_quesgen_bert.jsonnet --include-package nrl -s ../models/<quesgen-model-name> 

# Combine span-model and quesgen model into one model, which can then be run for prediction
python scripts/combine_models.py --span ./models/<span-model-name> --ques ../models/<quesgen-model-name> --out ../models/<full-model-name>.tar.gz
```
   
If you want to use the trained parsers from the Large Scale QA-SRL (2018) and the QANom (2020) papers, 
run `./qanom_parser/scripts/download_pretrained.sh`. This downloads both `qasrl_parser_elmo` and `qanom_parser_elmo` 
full models into `./models` directory (which is where we suggest to put your own models if you train any). 

### Inference 

To run prediction on new texts, you can use the `allennlp predict` command:
```bash
allennlp predict <model-dir-or-archive> <input-file> --include-package nrl --predictor qanom_parser --output-file <output-file>
```
This takes a JSON-lines <input-file>, with one line for each sentence, in the following format:
 ```json
{"qasrl_id": "Wiki1k:wikinews:1007169:1:0", "sentence": "She said in a statement : `` With an amazing portfolio of cars and trucks and the strongest financial performance in our recent history , this is an exciting time at today 's GM .", "predicate_indices": [4, 19], "verb_forms": ["state", "perform"]}
```
where "qasrl_id" is optional (and can be alternatively named "sentence_id" or "SentenceId"). 
Notice this input format requires more information than the `qasrl_parser` predictor 
(i.e., additional "predicate_indices" and "verb_forms" fields). 
This is because it expects a predicate detector module to pre-identify the nominal predicates 
for which it will generate QA annotations, along with their corresponding verbs.
Also note that no tokenizer model is applied on the sentence string - 
we assume the sentence is pre-tokenized (and joined with spaces).
  
The output-file will also be a JSON-lines file, in the following format:
```json
{
	"words": ["She", "said", "in", "a", "statement", ":", "``", "With", "an", "amazing", "portfolio", "of", "cars", "and", "trucks", "and", "the", "strongest", "financial", "performance", "in", "our", "recent", "history", ",", "this", "is", "an", "exciting", "time", "at", "today", "'s", "GM", "."],
	"verbs": [
		{
			"verb": "state",
			"qa_pairs": [
				{
					"question": "Who stated something?",
					"spans": [{"start": 0,"end": 0,"text": "She","score": 0.42914873361587527}],
					"slots": {"wh": "who","aux": "_","subj": "_","verb_slot_inflection": "Past","obj": "something","prep": "_","obj2": "_","is_passive": "False","is_negated": "False"}
				}
			],
			"index": 4
		},
		...
	],
	"qasrl_id": "Wiki1k:wikinews:1007169:1:0"
}
```

This is the same output format as of the QA-SRL parser, 
which is why predicates are called "verbs" even though for QANom they are nominal. 

#### Predicting from and to QANom CSV format
For running the QANom predictor on CSV-formatted input file - as those outputted by `predicate_detector`, 
with nominal predicate information (crucially, `target_idx` and `is_verbal` columns) - run:
```bash
python scripts/convert_csv_to_jsonl_input_for_predictor.py <qanom-predicate-data.csv>
```
This will generate a file in the JSON-lines format expected by `qanom_predictor`. 
The output file would have the same name as the input except for the file extension (`qanom-predicate-data.jsonl`).


To convert the predictor's output back into QANom's CSV format, run:
```bash
python scripts/convert_predictor_output_to_csv.py <predicted-qanom.jsonl>
```
Similarly, this would generate a `predicted-qanom.csv` file in a format equivalent to the QANom Dataset files.

## Evaluate
Given two QANom CSV files, e.g. `predicted.csv` and `gold.csv`, evaluate *predicted* with *gold* as reference using the command:
```bash
python qanom/evaluate predicted.csv gold.csv
```

