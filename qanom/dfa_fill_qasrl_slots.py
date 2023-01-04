from typing import List, Optional, Dict, Union

from constrained_decoding.dfa import DFA
from constrained_decoding.qasrl import get_qasrl_question_dfa

STATE_TO_SLOT = {
    0: "wh",
    1: "aux",
    2: "subj",
    3: "verb",
    "<1>_0": "obj",
    "<1>_1": "prep",
    "<1>_2": "obj2",
}

SLOT_TO_STATE = {slot: state for state, slot in STATE_TO_SLOT.items()}
Slots = Dict[str, str]

default_qasrl_question_dfa = get_qasrl_question_dfa(constrain_verb=False)

def dfa_fill_qasrl_slots(predicted_question: str, question_dfa: Optional[DFA] = None) -> Optional[Slots]:
    """
    Use `question_dfa` to fill QASRL slots from raw `predicted_question`.
    Empty slots in `predicted_question` is expected to be explicitly denoted by an underscore "_".
    """
    if question_dfa is None:
        question_dfa = default_qasrl_question_dfa
    lowered_question = predicted_question.lower()
    tokenized_question = lowered_question.split(" ")
    # handle '?' edge cases
    if tokenized_question[-1].endswith('?'):
        if not tokenized_question[-1] == '?':
            # seperate '?' to be the 8th slot when there is no space before it
            tokenized_question = tokenized_question[:-1] + [tokenized_question[-1][:-1], '?']
    else:
    # add '? as the 8th slot if non existent
        tokenized_question.append('?')
        
    slots: Optional[Slots] = _parse_token(tokenized_question, 0, {}, [], question_dfa)

    if slots:
        return slots


def _parse_token(tokenized_question, curr_state, slots: Slots, previous_tokens_of_slot: List[str], question_dfa: DFA) -> Optional[Slots]:
    if not(any(tokenized_question)):
        return None
    elif tokenized_question[0] == "?":
        if len(slots) == 7:
            return slots
        else:
            return None

    token = tokenized_question.pop(0)
    slot_filler_candidate = ' '.join(previous_tokens_of_slot + [token])
    success, next_state = question_dfa.step(curr_state, slot_filler_candidate)
    if success:
        # handle multi-word ambiguities and wildcard
        # token can be the slot, or complete the slot with previous_tokens_of_slot, but can also be only part of the slot
        
        # Recursive call according to possibility that this token exhoustes the current slot 
        slots_if_exhousted_slot = slots.copy()
        slots_if_exhousted_slot[STATE_TO_SLOT[curr_state]] = slot_filler_candidate
        slots_if_exhousted_slot = _parse_token(tokenized_question.copy(), next_state, slots_if_exhousted_slot, [], question_dfa)
        if slots_if_exhousted_slot:
            return slots_if_exhousted_slot
        else:
            # Recursive call according to possibility that this token has not yet exhoustes the current slot 
            return _parse_token(tokenized_question, curr_state, slots, previous_tokens_of_slot + [token], question_dfa)
        
    else:
        # Recursive call - might be that next tokens will complete current slot to a valid filler
        return _parse_token(tokenized_question, curr_state, slots, previous_tokens_of_slot + [token], question_dfa)
