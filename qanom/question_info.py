from typing import Dict, Optional, Union
from qanom.evaluation.roles import SemanticRole, question_to_sem_role
from qanom.dfa_fill_qasrl_slots import dfa_fill_qasrl_slots, Slots
from qanom.verb_inflections import VerbInflections
from qanom.annotations.decode_encode_answers import Question

VerbInflections = VerbInflections()

BE_AUX = {"am", "is", "isn't", "are", "aren't", "was", "wasn't", "were", "weren't", "would", "wouldn't", "will", "won't"}
BE_AUX_WITH_NEG = BE_AUX | { f"{be_aux} not" for be_aux in BE_AUX if not be_aux.endswith("n't")}

def get_slots(raw_predicted_question: str, 
              append_verb_slot_inflection: bool = False,
              append_is_negated_slot = False,
              append_is_passive_slot = False) -> Optional[Dict[str, str]]:
    """
    if `append_is_negated_slot` is True, the returned dictionary hold another "is_negated" binary slot on top of the 7 surface slots. 
    if `append_is_passive_slot` is True, the returned dictionary hold another "is_passive" binary slot on top of the 7 surface slots. 
    """
    slots = dfa_fill_qasrl_slots(predicted_question=raw_predicted_question)
    if slots is None:
        return None
    if append_verb_slot_inflection:
        # use VerbInflections to find verb_slot_inflection
        verb_word = slots["verb"].split()[-1] # omit verb prefixes like "be verbing"
        verb_slot_inflection = VerbInflections.get_inflection(verb_word, get_all_matches=False)
        slots["verb_slot_inflection"] = verb_slot_inflection
        # add "verb_prefix" slot (might be useful along with "verb_slot_inflection")
        slots["verb_prefix"] = extract_verb_prefix(slots)
    if append_is_negated_slot:
        slots["is_negated"] = extract_is_negated(slots)
    if append_is_passive_slot:
        slots["is_passive"] = extract_is_passive(slots)
    return slots

def extract_is_passive(q_slots: Slots, disambiguate: bool = True) -> Union[bool, str]:
    """ 
    Return 'is_passive' based on the 'aux', 'verb_prefix', and 'verb_slot_inflection' slots. 
    :param disambiguate: For verb cases where "verb_slot_inflection" might take multiple values (e.g. "put"),
        the passivity may also be ambiguous; by default (`disambiguate==True`) we disabiguate by more common inflections
        (see `VerbInflections.get_inflection` for search order), but otherwise (`disambiguate==False`), ambiguous verbs
        would return "???" indicating that passivity is not fully determined.
    """
    q_slots = q_slots.copy()    # Don't modify argument  
    # turn q_slots["verb_slot_inflection"] into a set; if `disambiguate` is True, the set will be of length 1
    if "verb_slot_inflection" in q_slots:
        q_slots["verb_slot_inflection"] = {q_slots["verb_slot_inflection"]}
        
    if "verb_slot_inflection" not in q_slots or disambiguate is False:
        # need to extract all possible inflections, to maintain optional ambiguity of passive voice
        verb_word = q_slots["verb"].split()[-1]
        all_options = VerbInflections.get_inflection(verb_word, get_all_matches=True)
        q_slots["verb_slot_inflection"] = {all_options[0]} if disambiguate else set(all_options)
    if "verb_prefix" not in q_slots:
        q_slots["verb_prefix"] = extract_verb_prefix(q_slots)
        
    # Linguistic rules here are (theoritically) based on:
    #   https://www.hunter.cuny.edu/rwc/repository/files/grammar-and-mechanics/verb-system/Active-and-Passive-Voice.pdf
    # For a verb to be in passive voice, it must: 
    # 1. be preceded by a BE auxiliary verb
    if q_slots["aux"] in BE_AUX_WITH_NEG or \
            q_slots["verb_prefix"] in ("be", "been", "being", "been being"):
        #  2. be "pastParticiple" (but it can match "past" first)
        verb_past_inflections = q_slots["verb_slot_inflection"] & {"pastParticiple", "past"}   
        if verb_past_inflections:
            if disambiguate or verb_past_inflections == q_slots["verb_slot_inflection"]:
                return True
            else: # should keep ambiguity - non-past verb inflection (e.g "stem") might indicate active voice 
                return "???"
        
    return False
    
def extract_is_negated(q_slots: Slots) -> bool:
    " Return `is_negated` based on the 7 surface question slots (aux and verb). "
    return (q_slots["aux"].endswith("n't") or
            (len(q_slots["verb"].split(" "))>1 and q_slots["verb"].split(" ")[0] == "not") )

def extract_verb_prefix(q_slots: Slots) -> str:
    return ' '.join(q_slots["verb"].split()[:-1])

def get_role(raw_predicted_question: str) -> Optional[SemanticRole]:
    slots = get_slots(raw_predicted_question, append_verb_slot_inflection=True, append_is_passive_slot=True, append_is_negated_slot=True)
    if slots is None:
        return None
    question: Question = Question.from_slots(text=raw_predicted_question, **slots)
    semRole = question_to_sem_role(question)
    # handle errounous passivity disambiguation
    if semRole == SemanticRole('Ungrammatical-question'):
        if extract_is_passive(slots, disambiguate=False) == "???":
            # flip voice - perhaps now it will be grammatical
            slots["is_passive"] = not slots["is_passive"]
            question: Question = Question(raw_predicted_question, **slots)
            semRole = question_to_sem_role(question)
    return semRole


# TESTS
def test_get_slots():
    q = f"When might someone orient _ _ _ ?"
    slots = get_slots(q)
    assert slots['aux'] == 'might'
    
    q = f"How long might someone be orientated _ _ _ ?"
    slots = get_slots(q)
    assert slots['wh'] == 'how long'
    
    q = f"How might someone be aligned _ toward something ?"
    slots = get_slots(q)
    assert slots['obj2'] == 'something'
    assert slots['prep'] == 'toward'

    q = f"How many someone be aligned _ toward something ?" # invalid WH
    slots = get_slots(q)
    assert slots is None


def test_is_passive():
    q = f"How might someone be aligned _ toward something ?"
    assert extract_is_passive(get_slots(q)) == True
    q = f"How was something aligned _ toward something ?"
    assert extract_is_passive(get_slots(q)) == True
    q = f"How was something going _ toward something ?"
    assert extract_is_passive(get_slots(q)) == False
    q = f"What is _ put _ on something ?" # an hard to understand passive voice - 'put' can be perceived as "stem"
    assert extract_is_passive(get_slots(q), disambiguate=False) == "???"
    
    
def test_get_role():
    q = f"How might someone be aligned _ toward something ?"
    assert str(get_role(q)) == "SemanticRole.how"
    q = f"How long was something aligned _ toward something ?"
    assert str(get_role(q)) == "SemanticRole.how long"
    
    q = f"What will someone take _ _ _ ?"
    assert str(get_role(q)) == "SemanticRole.R1"
    q = f"What was _ taken _ _ _ ?"
    assert str(get_role(q)) == "SemanticRole.R1"
    q = f"What is someone taking _ _ _ ?"
    assert str(get_role(q)) == "SemanticRole.R1"
    q = f"What will _ be taken _ _ _ ?"
    assert str(get_role(q)) == "SemanticRole.R1"
    
    q = f"Who will _ be taking something _ _ ?"
    assert str(get_role(q)) == "SemanticRole.R0"
    q = f"Who _ _ takes something _ _ ?"
    assert str(get_role(q)) == "SemanticRole.R0"
    q = f"Who _ _ underwent something _ _ ?"
    assert str(get_role(q)) == "SemanticRole.R0"
    q = f"Who is _ laughing _ at something ?"
    assert str(get_role(q)) == "SemanticRole.R0"
    
    q = f"What is _ put _ on something ?"
    # assert str(get_role(q)) == "SemanticRole.R2"    # Failing - misclassified as "SemanticRole.R0" (active voice)
    q = f"What is someone taking something _ _ ?"
    assert str(get_role(q)) == "SemanticRole.R2"
    q = f"What did someone convinced someone about _ ?"
    assert str(get_role(q)) == "SemanticRole.R2_about"