# QANom - Annotating Nominal Predicates with QA-SRL

This repository is the reference point for the data and software described in the paper 
*QANom: Question-Answer driven SRL for Nominalizations*.


## Dataset

The original QANom Dataset can be downloaded from this
 [google drive directory](https://drive.google.com/drive/folders/1MOA0Xkcz7Dezj5vJyq8YOM6lw85RR9xj).

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

### Module's pre-requisite 
```bash
pip install pandas nltk git+git://github.com/pattern3/pattern
```

### Usage
```bash
python qanom/candidate_extraction/candidate_extraction.py <input_file> <output_file> --read csv|josnl|raw --write csv|json [--no-wordnet] [--no-catvar] [--no-affixes]
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

The entry point is the script `candidate_extraction.py`. 

The module uses a POS tagger to `pos_tag` the sentence, and filter outs anything except common nouns (`get_common_nouns`). 
Then, it applies another filter - an "or" combination of two kinds of lexical-based filtering algorithms:
1. Lexical Resources based - Use WordNet & CatVar derivations. Any noun with verbal related derivation would be predicted as nom.
2. Affixes + seed based - Create a (possible-nominalization -> verb) list out of a verb seed list, using simple nominalization-suffixes substitution rules. 
Being in the list will be considered as being a nominalization candidate.

Filter 1 (`wordnet_util.py` + `catvar.py`) requires 
wordnet (available via [nltk](https://www.nltk.org/)) and [CatVar](https://clipdemos.umiacs.umd.edu/catvar/) 
(run `./scripts/download_catvar.sh` for downloading it into the `resources` directory).

Filter 2 (`verb_to_nom.py`) uses [pattern.en](https://www.clips.uantwerpen.be/pages/pattern-en) package (`pip install pattern3`).
 The package is not maintained by the authors and contain a few minor errors that is easy to fix on your local installation.  

If there are multiple derivationally related verbs, we select the verb that minimizes the edit distance with the noun. 


## QANom Predicate Detector
todo


## QANom Baseline parser 
The `qanom_parser` is essentially the [nrl-qasrl](https://github.com/kleinay/nrl-qasrl/tree/qanom) parser for QA-SRL, presented in 
[*Large-Scale QA-SRL Parsing* (FitzGerald et. al., 2018)](https://www.aclweb.org/anthology/P18-1191/).
To adapt the parser to QANom specifications (e.g. that the verb in the question is not the predicate itself) 
and format (csv), we have our own `qanom` branch on the `nrl-qasrl` repository. This branch uses the `qanom` package.
Run `./scripts/setup_parser.sh` to clone the parser into `qanom_parser` directory and prepare its prerequisites.
Then `cd qanom_parser` to run model-related commands as those described below.

### Training models
Follow the `README` in `qanom_parser` for instructions on training new QA-SRL models.

A QANom parser is trained using a CSV file (QANom format) as input, with the `QANomReader` DatasetReader 
(in `nrl/data/dataset_readers/qanom_reader.py`). For example: 

```bash
# first train a span-predictor for identifying answer spans (i.e. arguments)
allennlp train configs/train_qanom_span_elmo.jsonnet --include-package nrl -s data/trained_models/combined_span_elmo 

```
   
If you want to use the trained parsers from the Large Scale QA-SRL (2018) and the QANom (2020) papers, 
run `./qanom_parser/scripts/download_pretrained.sh`. This downloads both `qasrl_parser_elmo` and `qanom_parser_elmo` 
models into `./models` directory. If you train your own models, put them in this directory as well. 




