# QANom - Annotating Nominal Predicates with QA-SRL

QANom is a research project aiming for a natural representation of nominalization's predicate-argument relations.
It extends the Question Answer driven Semantic Role Labeling (QASRL) framework (see [website](http://qasrl.org/)), which tackled verbal predicates,
to the more challenging space of deverbal nominalizations. 

This repository is the reference point for the data and software described in the paper 
[*QANom: Question-Answer driven SRL for Nominalizations*](https://www.aclweb.org/anthology/2020.coling-main.274/) (COLING 2020).
To find information for replicating the work described by the QANom paper (crowdsourcing a QANom dataset, identifying nominalization candidates, training and evaluating the baseline models), please refer to the [paper_reference_readme.md](paper_reference_readme.md).

The repo also consists software for using QANom downstream. 
This mainly includes pipelines for easy usage of the [nominalization detection model](#nominalization-detection-model) and of the [QANom parsers](#qanom-sequence-to-sequence-model). 
This README will guide you through using this software. 


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

## End-to-End Pipeline 

If you wish to parse sentences with QANom, the best place to start is the `QANomEndToEndPipeline` class from the `qanom.qanom_end_to_end_pipeline` module.

This pipeline is first running the [Nominalization Detector](https://huggingface.co/kleinay/nominalization-candidate-classifier) for identifying the nominal predicates in the sentence (see [demo](https://huggingface.co/spaces/kleinay/nominalization-detection-demo)).
Then, it sends each nominal predicate to the [QAnom-Seq2Seq model](https://huggingface.co/kleinay/qanom-seq2seq-model-joint) (see [demo](https://huggingface.co/spaces/kleinay/qanom-seq2seq-demo)) to parse them with Question-Answer driven Semantic Role Labeling (QASRL).

**Usage Example**

```python
from qanom.qanom_end_to_end_pipeline import QANomEndToEndPipeline
pipe = QANomEndToEndPipeline(detection_threshold=0.75)
sentence = "The construction of the officer 's building finished right after the beginning of the destruction of the previous construction ."
print(pipe([sentence]))
```

Output:
```python
[[{'QAs': [{'question': 'what was constructed ?',
     'answers': ["the officer 's"]}],
   'predicate_idx': 1,
   'predicate': 'construction',
   'predicate_detector_probability': 0.7623529434204102,
   'verb_form': 'construct'},
  {'QAs': [{'question': 'what began ?',
     'answers': ['the destruction of the']}],
   'predicate_idx': 11,
   'predicate': 'beginning',
   'predicate_detector_probability': 0.8923847675323486,
   'verb_form': 'begin'},
  {'QAs': [{'question': 'what was destructed ?', 
     'answers': ['the previous']}],
   'predicate_idx': 14,
   'predicate': 'destruction',
   'predicate_detector_probability': 0.849774956703186,
   'verb_form': 'destruct'}]]
```

## Nominalization Detection Model

This model identifies "predicative nominalizations", that is, nominalizations that carry an eventive (or "verbal") meaning in context. It is a `bert-base-cased` pretrained model, fine-tuned for token classification on top of the "nominalization detection" task as defined and annotated by the QANom project.

The model is trained as a binary classifier, classifying candidate nominalizations. 
The candidates are extracted using a POS tagger (filtering common nouns) and additionally lexical resources (e.g. WordNet and CatVar), filtering nouns that have (at least one) derivationally-related verb. In the QANom annotation project, these candidates are given to annotators to decide whether they carry a "verbal" meaning in the context of the sentence. The current model reproduces this binary classification. 

Under the hood, the `NominalizationDetector` class encapsulates the full nominalization detection pipeline (i.e. candidate extraction + predicate classification).
It leverages the `qanom.candidate_extraction.candidate_extraction.py` module, and additionally downloads and wraps the [nominalization-candidate-classifier](https://huggingface.co/kleinay/nominalization-candidate-classifier) model, hosted at Huggingface model hub.

**Usage Example**

 ```python
from qanom.nominalization_detector import NominalizationDetector
detector = NominalizationDetector()
 
raw_sentences = ["The construction of the officer 's building finished right after the beginning of the destruction of the previous construction ."]

print(detector(raw_sentences, return_all_candidates=True))
print(detector(raw_sentences, threshold=0.75, return_probability=False))
```   

Outputs:
```python
[[{'predicate_idx': 1,
   'predicate': 'construction',
   'predicate_detector_prediction': True,
   'predicate_detector_probability': 0.7626778483390808,
   'verb_form': 'construct'},
  {'predicate_idx': 4,
   'predicate': 'officer',
   'predicate_detector_prediction': False,
   'predicate_detector_probability': 0.19832570850849152,
   'verb_form': 'officer'},
  {'predicate_idx': 6,
   'predicate': 'building',
   'predicate_detector_prediction': True,
   'predicate_detector_probability': 0.5794129371643066,
   'verb_form': 'build'},
  {'predicate_idx': 11,
   'predicate': 'beginning',
   'predicate_detector_prediction': True,
   'predicate_detector_probability': 0.8937646150588989,
   'verb_form': 'begin'},
  {'predicate_idx': 14,
   'predicate': 'destruction',
   'predicate_detector_prediction': True,
   'predicate_detector_probability': 0.8501205444335938,
   'verb_form': 'destruct'},
  {'predicate_idx': 18,
   'predicate': 'construction',
   'predicate_detector_prediction': True,
   'predicate_detector_probability': 0.7022264003753662,
   'verb_form': 'construct'}]]
```
```python
[[{'predicate_idx': 1, 'predicate': 'construction', 'verb_form': 'construct'},
  {'predicate_idx': 11, 'predicate': 'beginning', 'verb_form': 'begin'},
  {'predicate_idx': 14, 'predicate': 'destruction', 'verb_form': 'destruct'}]]
```
  

## QANom Sequence-to-Sequence Models 

We have finetuned T5, a pretrained Seq-to-Seq language model, on the task of parsing QANom QAs. 
Given a sentence and a highlighted nominal predicate, the models produce an output sequence consisting of the QANom-formatted question-answer pairs for this predicate.

We currently have two models:

* `qanom-seq2seq-model-baseline` ([HF repo](https://huggingface.co/kleinay/qanom-seq2seq-model-baseline)) - trained only on the QANom dataset. Performance: 57.6 Unlabled Arg F1, 34.9 Labeled Arg F1. 
* `qanom-seq2seq-model-joint` ([HF repo](https://huggingface.co/kleinay/qanom-seq2seq-model-joint)) - trained jointly on the QANom and verbal QASRL. Performance: 60.1 Unlabled Arg F1, 40.6 Labeled Arg F1. 

We provide the `QASRL_Pipeline` class (at `qanom.qasrl_seq2seq_pipeline) which is a Huggingface Pipeline for applying the models out-of-the-box on new texts:

```python
from pipeline import QASRL_Pipeline
pipe = QASRL_Pipeline("kleinay/qanom-seq2seq-model-baseline")
pipe("The student was interested in Luke 's <predicate> research about see animals .", verb_form="research", predicate_type="nominal")
``` 
Which will output:
```python
[{'generated_text': 'who _ _ researched something _ _ ?<extra_id_7> Luke', 
  'QAs': [{'question': 'who researched something ?', 'answers': ['Luke']}]}]
```   
You can learn more about using `transformers.pipelines` in the [official docs](https://huggingface.co/docs/transformers/main_classes/pipelines).

Notice that you need to specify which word in the sentence is the predicate, about which the question will interrogate. By default, you should precede the predicate with the `<predicate>` symbol, but you can also specify your own predicate marker:
```python
pipe("The student was interested in Luke 's <PRED> research about see animals .", verb_form="research", predicate_type="nominal", predicate_marker="<PRED>")
```
In addition, you can specify additional kwargs for controling the model's decoding algorithm:
```python
pipe("The student was interested in Luke 's <predicate> research about see animals .", verb_form="research", predicate_type="nominal", num_beams=3)
```


## Cite

 ```latex
 @inproceedings{klein2020qanom,
  title={QANom: Question-Answer driven SRL for Nominalizations},
  author={Klein, Ayal and Mamou, Jonathan and Pyatkin, Valentina and Stepanov, Daniela and He, Hangfeng and Roth, Dan and Zettlemoyer, Luke and Dagan, Ido},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={3069--3083},
  year={2020}
}
 ``` 
