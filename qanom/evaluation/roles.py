"""
Reconstructing the He et. al (2015) mapping of QA-SRL questions to Roles (See Section 4 in their paper).
They define the role space (R) as:
    • R = {R0, R1, R2, R2[p], w, w[p]}
    • w isin {Where, When, Why, How, HowMuch}
    • p isin prepositions

"""
import enum
import logging

from qanom.annotations.decode_encode_answers import Question

# create logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

core_wh_words = {'what', 'who'}
adjunct_wh_words = {'how', 'how long', 'how much', 'when', 'where', 'why'}

all_prepositions_in_dataset = {
 'about',
 'against',
 'along',
 'around',
 'as',
 'as doing',
 'at',
 'between',
 'by',
 'by doing',
 'doing',
 'during',
 'for',
 'for doing',
 'from',
 'in',
 'in doing',
 'into',
 'of',
 'of doing',
 'on',
 'on doing',
 'out',
 'over',
 'to',
 'to do',
 'to doing',
 'towards',
 'with',
 'up to',
 'with doing'
}|{  # here O added some thta are in QANom dataset so that it will work also on QASRL-GS
     '',
     'across',
     'after',
     'ahead',
     'ahead at',
     'along with',
     'amid',
     'among',
     'aside',
     'aside for',
     'before',
     'behind',
     'do',
     'down',
     'down from',
     'from among',
     'from doing',
     'in from',
     'in to',
     'off',
     'off for',
     'off from',
     'off of',
     'off to',
     'on to',
     'on to do',
     'onto',
     'out by',
     'out of',
     'out of doing',
     'out to',
     'out to do',
     'over from',
     'through',
     'to as',
     'under',
     'up',
     'up doing',
     'up for',
     'up with',
     'upon',
     'without'}

UNGRAMMATICAL = "Ungrammatical-question" # this role is for predicted question with non-grammatical structure
"""
SemanticRole will represented by a string, denoting the roles by:
    "R0", "R1", "R2", "where", "when", ..., "R2_about", "R2_against", ..., "where_about", "where_against", ... 
"""
role_space = {"R0", "R1", "R2"} | \
             adjunct_wh_words | \
             {"R2_" + p
              for p in all_prepositions_in_dataset} | \
             {wh + "_" + p
              for wh in adjunct_wh_words
              for p in all_prepositions_in_dataset} | \
             {UNGRAMMATICAL} # this role is for predicted question with non-grammatical structure

SemanticRole = enum.Enum("SemanticRole", {r: r for r in role_space})


def is_equivalent_question(q1: Question, q2: Question) -> bool:
    return question_to_sem_role(q1) == question_to_sem_role(q2)


def question_to_sem_role(q: Question) -> SemanticRole:
    """ Refer to Table 7 at He et. al. (2015). """
    def get_R2() -> SemanticRole:
        # distinguish direct object-2 or indirect object-2 by presence of preposition
        if q.prep:
            return SemanticRole("R2_" + q.prep)
        else:
            return SemanticRole.R2

    if q.wh.lower() in core_wh_words and not q.is_passive:
        if not q.subj:
            return SemanticRole.R0
        elif not q.obj:
            return SemanticRole.R1
        elif not q.obj2 or q.obj2 in ['do', 'doing']:
            return get_R2()
        else:
            logger.warning(f"ungrammatical question: {q} ; all core args are present in core-role active question.")
            return SemanticRole(UNGRAMMATICAL)

    elif q.wh.lower() in core_wh_words and q.is_passive:
        if not q.subj:
            return SemanticRole.R1
        elif not q.obj:
            # prep=="by" is an exception
            if q.prep=="by":
                return SemanticRole.R0
            else:
                return get_R2()
        else:
            # not in their table, but can be found in data - e.g. "what was someone awarded something for?"
            if q.prep == "by":
                return SemanticRole.R0
            else:
                return get_R2()

    # voice (passive/active) has no effect on adjunct questions
    elif q.wh.lower() in adjunct_wh_words:
        if not q.obj2 and q.prep:
            return SemanticRole(q.wh + "_" + q.prep)
        else:
            return SemanticRole(q.wh)

    logger.warning(f"ungrammatical question: {q}")
    return SemanticRole(UNGRAMMATICAL)

