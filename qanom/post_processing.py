"""
A script with a set of post-processing utilities for the collected annotation files (CSVs).
"""
from typing import NoReturn

import json
import pandas as pd

from qanom import prepare_nom_ident_batch, prepare_qanom_prompts


def find_invalid_prompts(annot_df : pd.DataFrame) -> pd.DataFrame:
    """
    Verify that all the HITs of the annotated data are valid according to the latest version of
    prepare_nom_ident_batch.py, where I have found and corrected errors about:
        * Wordnet edit-distance threshold
        * Wordnet verbalize() algorithm - using lemmas instead of synsets (large impact!)
    The function mark all rows with an (currently) invalid prompt (i.e. wrong verb_form for the target noun).
    One should leverage the returned CSV to generate a new JSON for re-annotating the invalid prompts with
    the corrected ones. Use
    :param annot_df:
    :return: a DataFrame adding to annot_df:
        * an "invalid_prompt" column (boolean) - True if prompt changed (requires re-annotation or removal).
        * a "corrected_verb_form" column (string) indicating the up-to-date suggested verb_form returned by
        prepare_nom_ident_batch.py. Notice that if it's an empty string, it means that for the current algorithm,
        no verb_form is available for this noun, meaning that we should delete the current annotation row but we should
        not re-annotate this target noun.
    """
    def annot_row_to_corrected_verb_form(row: pd.Series) -> str:
        noun = row.verb
        uptodate_verb_forms, is_had_uptodate_verbs = \
            prepare_nom_ident_batch.get_verb_forms_from_lexical_resources(noun, **prepare_qanom_prompts.resources)
        # return empty string for non-verbal nouns (nouns with no verb_forms)
        if not is_had_uptodate_verbs:
            return ""
        else:
            # take the best verb_form
            return uptodate_verb_forms[0]

    # use annot_row_to_corrected_verb_form to generate the new columns
    annot_df["corrected_verb_form"] = annot_df.apply(annot_row_to_corrected_verb_form, axis=1)
    annot_df["invalid_prompt"] = annot_df.apply(lambda row: row["corrected_verb_form"] != row["verb_form"])
    return annot_df


def reannotate_corrected_verb_forms(annot_df: pd.DataFrame, output_json_fn) -> NoReturn:
    """
    Generate input for the qasrl_crowdsourcing project (equivalent to output of prepare_qanom_prompts.py)
    :param annot_df: returned from find_invalid_prompts()
    :param output_json_fn: file name where to dump the JSON of the prompts for re-annotation
    """
    annot_df.drop_duplicates(subset=["qasrl_id", "verb_idx"])
    annot_df = annot_df[annot_df["invalid_prompt"]]    # filter only invalidated prompts
    candidates = [{"sentenceId": row["qasrl_id"],
                   "tokSent" : row["sentence"].split(" "),
                   "targetIdx": row["verb_idx"],
                   "verbForms": [row["corrected_verb_form"]]}
                  for id, row in annot_df.iterrows()]
    json.dump(candidates, open(output_json_fn, "w"))

