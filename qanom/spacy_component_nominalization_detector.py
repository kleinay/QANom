import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from transformers import AutoModelForTokenClassification, AutoTokenizer
from qanom.nominalization_detector import NominalizationDetector
from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources

default_model_hub_name = "kleinay/nominalization-candidate-classifier"

@Language.factory("nominalization_detector", default_config={"threshold": 0.7, "device": -1})
def create_nominalization_detector_component(nlp: Language, name: str, threshold: float, device: int, classifier_model_hub_name: str = None):
    return NominalizationDetectorComponent(nlp, threshold=threshold, device=device, model_hub_name=classifier_model_hub_name)

class NominalizationDetectorComponent:
    def __init__(self, nlp: Language, threshold: float, device: int, model_hub_name: str = None):
        # explicitly use candidate extraction and predicate detector separately
        self.model_hub_name = model_hub_name or default_model_hub_name
        self.candidate_classifier = AutoModelForTokenClassification.from_pretrained(self.model_hub_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_hub_name)
        self.device = f"cuda:{device}" if device>=0 else "cpu" # represent device in pytorch convention 
        self.threshold = threshold
        # set extensions on spacy classes
        Doc.set_extension("nominalizations", default=[], force=True)
        Token.set_extension("is_candidate_nominalization", default=False, force=True)
        Token.set_extension("is_nominalization", default=False, force=True)
        Token.set_extension("is_nominalization_confidence", default=None, force=True)
        Token.set_extension("verb_form", default=None, force=True)

    def __call__(self, doc: Doc) -> Doc:
        
        # Add `is_candidate_nominalization` and `is_nominalization` attribute to tokens
        for token in doc:
            token._.is_candidate_nominalization = False
            token._.is_nominalization = False
            if token.tag_.startswith("NN"):
                verb_forms, is_candidate = get_verb_forms_from_lexical_resources(token.text)
                if is_candidate:
                    verb_form = verb_forms[0]
                    token._.is_candidate_nominalization = True
                    token._.verb_form = verb_form
        # Apply classifier at sentence level
        #  Prepare Tokenization
        tokenized = self.tokenizer([sent.text for sent in doc.sents], return_tensors="pt", padding=True).to(self.device)
        # Run binary classification model on candidate nouns
        model = self.candidate_classifier.to(self.device)
        logits = model(tokenized.input_ids).logits
        # compose binary probability as the mean of positive label's prob with complement of neg. label's prob.
        positive_lbl_id = 0     # because label_map[0] == 'True' 
        negative_lbl_id = 1     # because label_map[1] == 'False'
        positive_probability = logits[:,:,positive_lbl_id].sigmoid()   # probability of the "true" label (index 0 in axis-2)
        negative_probability = logits[:,:,negative_lbl_id].sigmoid()
        probability = (positive_probability + (1-negative_probability) ) / 2
        preds = probability > self.threshold
        # iterate over tokens and set `is_nominalization` and `is_nominalization_confidence` attributes for candidate nouns
        nominalizations = []
        for i, sent in enumerate(doc.sents):
            #  get mapping of word to tokens (for using indexes from dataset)
            wordId2numOfTokens = [len(self.tokenizer.tokenize(t.text)) for t in sent]
            wordId2firstTokenId, curr_tok_id = [], 1
            for wordId in range(len(wordId2numOfTokens)):
                wordId2firstTokenId.append(curr_tok_id)
                curr_tok_id += wordId2numOfTokens[wordId]
            for token_i, token in enumerate(sent):
                if token._.is_candidate_nominalization:
                    token_id = wordId2firstTokenId[token_i]
                    token._.is_nominalization = preds[i][token_id].item()
                    token._.is_nominalization_confidence = probability[i][token_id].item()
                    if token._.is_nominalization:
                        nominalizations.append(token)
                
        # Add `nominalizations` attribute to doc
        doc._.nominalizations = nominalizations
    
        return doc
    
# Example usage:
# *************************
# nlp = spacy.load("en_core_web_sm")
# nlp.add_pipe("nominalization_detector", after="tagger", config={"threshold": 0.7, "device": -1})
# doc = nlp("The destruction of the building was caused by the earthquake.")
# print([(t.text, t._.is_nominalization, t._.is_nominalization_confidence) for t in doc if t._.is_candidate_nominalization])
# print(doc._.nominalizations)
# -----------------------------
# [('destruction', True, 0.8630461692810059), ('building', False, 0.5001491904258728)]
# [destruction]

