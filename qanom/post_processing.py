"""
A script with a set of post-processing utilities for the collected annotation files (CSVs).
"""
import json
import os.path
from typing import *

import pandas as pd

from qanom import prepare_qanom_prompts


def find_invalid_prompts(annot_df : pd.DataFrame) -> pd.DataFrame:
    """
    Verify that all the HITs of the annotated data are valid according to the latest version of
    prepare_qanom_prompts.py, where I have found and corrected errors about:
        [* Wordnet edit-distance threshold] (deprecated after next change)
        * Wordnet verbalize() algorithm - using lemmas instead of synsets (large impact!)
        * Tokenization issues - pos_tagger imposes different tokenization from sentence.split() or nltk.tokenize;
            in final version I'm using stanford-core-nlp pos_tagger to tokenize.

    The function mark all rows with an (currently) invalid prompt (i.e. wrong verb_form for the target noun).
    One should leverage the returned CSV to generate a new JSON for re-annotating the invalid prompts with
    the corrected ones. Use
    :param annot_df:
    :return: a DataFrame adding to annot_df:
        * an "invalid_prompt" column (boolean) - True if prompt changed (requires re-annotation or removal).
        * a "corrected_verb_form" column (string) indicating the up-to-date suggested verb_form returned by
        prepare_qanom_prompts.py. Notice that if it's an empty string, it means that for the current algorithm,
        no verb_form is available for this noun, meaning that we should delete the current annotation row but we should
        not re-annotate this target noun.
    """
    def annot_row_to_corrected_verb_form(row: pd.Series) -> str:
        noun = row.verb
        uptodate_verb_forms, is_had_uptodate_verbs = \
            prepare_qanom_prompts.get_verb_forms_from_lexical_resources(noun, **prepare_qanom_prompts.resources)
        # return empty string for non-verbal nouns (nouns with no verb_forms)
        if not is_had_uptodate_verbs:
            return ""
        else:
            # take the best verb_form
            return uptodate_verb_forms[0]

    # use annot_row_to_corrected_verb_form to generate the new columns
    annot_df["corrected_verb_form"] = annot_df.apply(annot_row_to_corrected_verb_form, axis=1)
    annot_df["invalid_prompt"] = annot_df.apply(lambda row: row["corrected_verb_form"] != row["verb_form"], axis=1)
    return annot_df


def reannotate_corrected_verb_forms(annot_df: pd.DataFrame, output_json_fn) -> NoReturn:
    """
    Generate input for the qasrl_crowdsourcing project (equivalent to output of prepare_qanom_prompts.py)
    :param annot_df: returned from find_invalid_prompts()
    :param output_json_fn: file name where to dump the JSON of the prompts for re-annotation
    """
    annot_df.drop_duplicates(subset=["qasrl_id", "verb_idx"])
    invalidated_df = annot_df[annot_df["invalid_prompt"]]    # filter only invalidated prompts
    re_annot_df = invalidated_df[invalidated_df["corrected_verb_form"]!=""] # filter out predicates now with no verb form
    candidates = [{"sentenceId": row["qasrl_id"],
                   "tokSent" : row["sentence"].split(" "),
                   "targetIdx": row["verb_idx"],
                   "verbForms": [row["corrected_verb_form"]]}
                  for id, row in re_annot_df.iterrows()]
    json.dump(candidates, open(output_json_fn, "w"))


def replace_some_annotations(orig_big_df: pd.DataFrame, corrected_annot_df: pd.DataFrame) -> pd.DataFrame:
    """  Return orig_big_df after replacing the predicates that are in corrected_annot_df with their
         corrected annotation. """
    # replace intersection of predicates
    predicates_to_replace = set(corrected_annot_df.key.drop_duplicates()) & set(orig_big_df.key.drop_duplicates())
    # remove them from (a copy of) orig
    final_df = orig_big_df[~orig_big_df.key.isin(predicates_to_replace)].copy()
    # now merge with the annotations of these predicates in corrected_annot_df
    relevant_corrected_df = corrected_annot_df[corrected_annot_df.key.isin(predicates_to_replace)].copy()
    final_df = pd.concat([final_df, relevant_corrected_df], sort=False)
    return final_df


def fix_annot_with_corrected(orig_annot_fn: str, corrected_annot_fn: str,
                             dest_dir: str = "files/annotations/production/corrected") -> NoReturn:
    from annotations_evaluations.common import read_annot_csv, save_annot_csv
    orig_df = read_annot_csv(orig_annot_fn)
    all_corrected_df = read_annot_csv(corrected_annot_fn)
    corrected_df = replace_some_annotations(orig_df, all_corrected_df)
    # now export to file with same naming as orig (but in destination folder)
    orig_dir, orig_name = os.path.split(orig_annot_fn)
    dest_fn = os.path.join(dest_dir, orig_name)
    save_annot_csv(corrected_df, dest_fn)
