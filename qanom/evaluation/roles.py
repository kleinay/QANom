"""
Reconstructing the He et. al (2015) mapping of QA-SRL questions to Roles (See Section 4 in their paper).
They define the role space (R) as:
    • R = {R0, R1, R2, R2[p], w, w[p]}
    • w isin {Where, When, Why, How, HowMuch}
    • p isin prepositions

"""
import enum

from annotations.decode_encode_answers import Question

core_wh_words = {'what', 'who'}
adjunct_wh_words = {'how', 'how long', 'how much', 'when', 'where', 'why'}

all_prepositions_in_gold = {
 'about',
 'against',
 'along',
 'as',
 'as doing',
 'at',
 'between',
 'by',
 'by doing',
 'doing',
 'for',
 'from',
 'in',
 'into',
 'of',
 'of doing',
 'on',
 'on doing',
 'over',
 'to',
 'to do',
 'to doing',
 'towards',
 'with'
}


"""
SemanticRole will represented by a string, denoting the roles by:
    "R0", "R1", "R2", "where", "when", ..., "R2_about", "R2_against", ..., "where_about", "where_against", ... 
"""
role_space = {"R0", "R1", "R2"} | \
             adjunct_wh_words | \
             {"R2_" + p
              for p in all_prepositions_in_gold} | \
             {wh + "_" + p
              for wh in adjunct_wh_words
              for p in all_prepositions_in_gold}

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

    if q.wh in core_wh_words and not q.is_passive:
        if not q.subj:
            return SemanticRole.R0
        elif not q.obj:
            return SemanticRole.R1
        elif not q.obj2:
            return get_R2()
        else:
            raise Exception(f"ungrammatical question: {q} ; all core args a present in core-role active question.")

    elif q.wh in core_wh_words and q.is_passive:
        if not q.subj:
            return SemanticRole.R1
        elif not q.obj:
            # prep=="by" is an exception
            if q.prep=="by":
                return SemanticRole.R0
            else:
                return get_R2()
        else:
            raise Exception(f"ungrammatical question: {q} ; both subj and obj are present in core-role passive question.")

    # voice (passive/active) has no effect on adjunct questions
    elif q.wh in adjunct_wh_words:
        if not q.obj2 and q.prep:
            return SemanticRole(q.wh + "_" + q.prep)
        else:
            return SemanticRole(q.wh)

    raise Exception(f"ungrammatical question: {q}")
