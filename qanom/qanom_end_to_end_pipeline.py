from typing import Iterable, Optional

from qanom.nominalization_detector import NominalizationDetector
from qanom.qasrl_seq2seq_pipeline import QASRL_Pipeline

qanom_models = {"baseline": "kleinay/qanom-seq2seq-model-baseline",
                "joint": "kleinay/qanom-seq2seq-model-joint"}  

default_detection_threshold = 0.7
default_model = "joint"

class QANomEndToEndPipeline():
    """
    This pipeline wraps the NominalizationDetector together with the Seq2Seq-based QANom parser.
    Given a sentence, the pipeline identifies nominal predicates (deverbal nominalizations), 
    and generates question-answer pairs capturing their predicate-argument relations.  
    """
    
    def __init__(self, 
                 qanom_model: Optional[str] = None, 
                 detection_threshold: Optional[float] = None):
        self.predicate_detector = NominalizationDetector()
        self.detection_threshold = detection_threshold or default_detection_threshold
        
        qanom_model = qanom_model or default_model
        model_url = qanom_models[qanom_model] if qanom_model in qanom_models else qanom_model
        self.qa_pipeline = QASRL_Pipeline(model_url)
    
    def __call__(self, sentences: Iterable[str], 
                 detection_threshold = None,
                 return_detection_probability = True,
                 **generate_kwargs):
        # get predicates
        threshold = detection_threshold or self.detection_threshold
        predicate_infos_for_sentences = self.predicate_detector(sentences, 
                                                                threshold=threshold,
                                                                return_probability=return_detection_probability)
        outputs = []
        for sentence, predicate_infos in zip(sentences, predicate_infos_for_sentences):
            # collect QAs for all predicates in sentence 
            predicates_full_infos = []
            for pred_info in predicate_infos:
                model_input = self._prepare_input_sentence(sentence, pred_info['predicate_idx'])
                model_output = self.qa_pipeline(model_input, 
                                                verb_form=pred_info['verb_form'], 
                                                predicate_type="nominal",
                                                **generate_kwargs)
                predicates_full_info = dict(QAs=model_output['QAs'], **pred_info)
                predicates_full_infos.append(predicates_full_info)
            outputs.append(predicates_full_infos)
        return outputs
    
    def _prepare_input_sentence(self, raw_sentence: str, predicate_idx: int) -> str:
        words = raw_sentence.split(" ") 
        words = words[:predicate_idx] + ["<predicate>"] + words[predicate_idx:] 
        return " ".join(words)      
        

if __name__ == "__main__":
    pipe = QANomEndToEndPipeline(detection_threshold=0.75)
    sentence = "The construction of the officer 's building finished right after the beginning of the destruction of the previous construction ."
    print(pipe([sentence]))
    # Output:
    # [[{'QAs': [{'question': 'what was constructed ?',
    #      'answers': ["the officer 's"]}],
    #    'predicate_idx': 1,
    #    'predicate': 'construction',
    #    'predicate_detector_probability': 0.7623529434204102,
    #    'verb_form': 'construct'},
    #   {'QAs': [{'question': 'what began ?',
    #      'answers': ['the destruction of the']}],
    #    'predicate_idx': 11,
    #    'predicate': 'beginning',
    #    'predicate_detector_probability': 0.8923847675323486,
    #    'verb_form': 'begin'},
    #   {'QAs': [{'question': 'what was destructed ?', 
    #      'answers': ['the previous']}],
    #    'predicate_idx': 14,
    #    'predicate': 'destruction',
    #    'predicate_detector_probability': 0.849774956703186,
    #    'verb_form': 'destruct'}]]