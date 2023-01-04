"""
Handle Wiktionary data ("qanom/resources/wiktionary/en_verb_inflections.txt") - enable:
    * Get all inflected forms of a verb
    * Find the inflection of the verb mention (heuristically by search)
"""
from typing import Dict, Optional, Union, List
from pathlib import Path
import pandas as pd
from qanom.config import resources_path

wiktionary_inflections_data_fn = Path(resources_path) / "wiktionary" / "en_verb_inflections.txt"
wiktionary_inflections_columns = ["stem", "presentSingular3rd", "presentParticiple", "past", "pastParticiple"] 
# column names are taken from https://github.com/julianmichael/jjm/blob/master/jjm/src-jvm/ling/en/Inflections.scala

class VerbInflections():
    def __init__(self) -> None:
        self.inflections_df = pd.read_csv(wiktionary_inflections_data_fn, sep="\t", names=wiktionary_inflections_columns)
        
    def get_inflected_forms(self, verb: str, get_all_matches: bool = False) -> Optional[Dict[str, str]]:
        """ 
        Search the verb in wiktionary data, and return a dict describing the 5 inflected-forms of this verb stem.
        Searching column by column, following `wiktionary_inflections_columns` order. 
        :param get_all_matches: Whether to return a list of all possible inflection interpretations for this verb-form.
            Otherwise, return the first according to search order. 
        """
        all_inflections = []
        for col in wiktionary_inflections_columns:
            optional_row = self.inflections_df[self.inflections_df[col]==verb]
            inflections = optional_row.to_dict('records')
            all_inflections.extend(inflections)

        if all_inflections:
            if get_all_matches:
                return all_inflections
            else:
                return all_inflections[0]

    def get_inflection(self, verb: str, get_all_matches: bool) -> Optional[Union[str, List[str]]]:
        """
        Return the inflection label of `verb` (out of `wiktionary_inflections_columns`). 
        Search order is defined by `wiktionary_inflections_columns`.
        :param get_all_matches: Whether to return a list of all possible inflection interpretations for this verb-form.
            Otherwise, return the first according to search order. 
        """
        inflection_form_dicts = self.get_inflected_forms(verb, get_all_matches=True)
        forms_of_this_verb_mention = []
        if inflection_form_dicts:
            for inflection_forms in inflection_form_dicts:
                for inflection_name in wiktionary_inflections_columns:
                    if inflection_forms[inflection_name] == verb:
                        forms_of_this_verb_mention.append(inflection_name)
            if get_all_matches:
                return forms_of_this_verb_mention
            else:
                return forms_of_this_verb_mention[0]
        
