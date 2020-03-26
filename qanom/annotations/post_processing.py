"""
A script with a set of post-processing utilities for the collected annotation files (CSVs).
"""
import json
import os.path
from typing import *

import pandas as pd

from qanom import utils
from qanom.annotations import common


# generic post-processing function
def postprocess_annotation_files(orig_dir: str, dest_dir: str,
                                 process_annot_func: Callable[[pd.DataFrame,], pd.DataFrame],
                                 file_name_modification_func: Callable[[str,], str] = lambda s:s) -> NoReturn:
    """
    :param orig_dir: Directory from which to take the annottion to process (input)
    :param dest_dir: Directory to which the processed annotation files are to be exported
    :param process_annot_func: a function that gets an annot_df and returns a processed (i.e. corrected or changed,
    to some aspect) annot_df
    :param file_name_modification_func: how to change an annotation file-name from source-dir to dest-dir
    :return:
    """
    ann_files = [os.path.join(orig_dir, fn) for fn in os.listdir(orig_dir) if fn.endswith(".csv")]
    from annotations.common import read_annot_csv, save_annot_csv
    for orig_fn in ann_files:
        orig_df = read_annot_csv(orig_fn)
        new_df = process_annot_func(orig_df)
        # now export to file with same naming as orig (but in destination folder)
        orig_dir, orig_name = os.path.split(orig_fn)
        new_name = file_name_modification_func(orig_name)
        dest_fn = os.path.join(dest_dir, new_name)
        save_annot_csv(new_df, dest_fn)
        print(f"exported annotations to {dest_fn}")


def merge_csvs(csvs: List[str], dest: str) -> pd.DataFrame:
    import qanom.annotations.common as common
    dfs = [common.read_csv(csv_fn) for csv_fn in csvs]
    merged_df = df=pd.concat(dfs, ignore_index=True, sort=False)
    merged_df.to_csv(dest, index=False, encoding="utf-8")
    print(f"exported DataFrame with shape {merged_df.shape} to {dest}")
    return merged_df


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
        from candidate_extraction import prepare_qanom_prompts
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


def replace_some_annotations(orig_df: pd.DataFrame, corrected_annot_df: pd.DataFrame) -> pd.DataFrame:
    """  Return orig_df after replacing the predicates that are in corrected_annot_df with their
         corrected annotation. """
    # replace intersection of predicates
    predicates_to_replace = set(corrected_annot_df.key.drop_duplicates()) & set(orig_df.key.drop_duplicates())
    # remove them from (a copy of) orig
    final_df = orig_df[~orig_df.key.isin(predicates_to_replace)].copy()
    # now merge with the annotations of these predicates in corrected_annot_df
    relevant_corrected_df = corrected_annot_df[corrected_annot_df.key.isin(predicates_to_replace)].copy()
    final_df = pd.concat([final_df, relevant_corrected_df], sort=False)
    return final_df


def fix_annot_with_corrected(orig_annot_fn: str, corrected_annot_fn: str,
                             dest_dir: str = "files/annotations/production/corrected") -> NoReturn:
    from annotations.common import read_annot_csv, save_annot_csv
    orig_df = read_annot_csv(orig_annot_fn)
    all_corrected_df = read_annot_csv(corrected_annot_fn)
    corrected_df = replace_some_annotations(orig_df, all_corrected_df)
    # in addition to re-annotation correction, filter out currently invalid prompts for data
    corrected_df = find_invalid_prompts(corrected_df)
    corrected_and_filtered_df = corrected_df[~corrected_df.invalid_prompt]
    final_df = corrected_and_filtered_df.drop(["corrected_verb_form", "invalid_prompt"], axis=1)
    # now export to file with same naming as orig (but in destination folder)
    orig_dir, orig_name = os.path.split(orig_annot_fn)
    dest_fn = os.path.join(dest_dir, orig_name)
    save_annot_csv(final_df, dest_fn)


def generate_corrected_annotation_files():
    """
    Fixing original annotations with new verb-form algorithm and the resulting output of the re-annotation batch.
    """
    orig_dir_path = "files/annotations/production"
    cor_path = "files/annotations/production/corrected"
    reannot_fn = "files/annotations/reannot.csv"
    ann_files = [os.path.join(orig_dir_path, fn) for fn in os.listdir(orig_dir_path) if fn.endswith(".csv")]
    for orig_fn in ann_files:
        fix_annot_with_corrected(orig_fn, reannot_fn, cor_path)

"""
Functions for filtering Non-Morpholigically-Related verb_forms (based on NMR feedbacks from workers)
Note: the methods that fix the annotations use the file files/nmr_case.csv.  This file 
 is a black-list containing all (noun, verb_form) pairs that should be removed, and it can be edited.
"""
nmr_cases_fn = 'files/nmr_cases.csv'


def generate_filtered_annotation_files():
    """
    Filtering annotations by removing NMR (noun, verb_form) cases.
    """
    # orig_dir_path = "files/annotations/gold_set/generation/corrected"
    # dest_path = "files/annotations/gold_set/generation/corrected_filtered"
    orig_dir_path = "files/annotations/train_set/orig"
    dest_path = "files/annotations/train_set/filtered"
    ann_files = [os.path.join(orig_dir_path, fn) for fn in os.listdir(orig_dir_path) if fn.endswith(".csv")]
    for orig_fn in ann_files:
        fix_annot_with_nmr_blacklist(orig_fn, dest_path)


def fix_annot_with_nmr_blacklist(orig_annot_fn: str,
                                 dest_dir: str) -> NoReturn:
    from annotations.common import read_annot_csv, save_annot_csv
    orig_df = read_annot_csv(orig_annot_fn)
    filtered_df = remove_NMR_cases_from_annotations(orig_df)
    # now export to file with same naming as orig (but in destination folder)
    orig_dir, orig_name = os.path.split(orig_annot_fn)
    dest_fn = os.path.join(dest_dir, orig_name)
    save_annot_csv(filtered_df, dest_fn)


def remove_NMR_cases_from_annotations(annot_df: pd.DataFrame) -> pd.DataFrame:
    """ Take all predicates with NMR feedbacks, and remove them. return the filtered df. """
    import csv
    with open(nmr_cases_fn) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        not_morphologically_related_cases = list(csv_reader)[1:]
    # incorporate both capital and non-capital to black-list of NMRs
    nmrs = set(utils.flatten([[(noun.capitalize(), verb),
                        (noun.casefold(), verb)]
                        for noun,verb in not_morphologically_related_cases]))
    # remove these from annot_df
    is_nmr = lambda r: (r.verb, r.verb_form) in nmrs
    is_nmr_series =  annot_df.apply(is_nmr, axis=1)
    print(f"filtering out {annot_df[is_nmr_series]['key'].drop_duplicates().shape[0]} Non-Morphologically-Related predicates ")
    filtered_df = annot_df[~is_nmr_series]
    return filtered_df


def set_sentence_column(df: pd.DataFrame, df_with_sentences: pd.DataFrame) -> NoReturn:
    """ Add 'sentence' to df by qasrl_id (inplace).
    Use df_with_sentences (assuming it has both qasrl_id and sentence columns) to take the sentence from. """
    sent_id2sent = dict(zip(df_with_sentences.qasrl_id, df_with_sentences.sentence))
    df['sentence'] = df.qasrl_id.apply(sent_id2sent.get)


def get_NMR_cases(feedbacks_dir: str) -> Set[Tuple[str, str]]:
    """ get all NMR cases - all (noun, verb_form) pairs that got any NMR feedback anywhere.
     The function would work for feedback files containing 'sentence' column.
     """
    from annotations.common import read_dir_of_csv
    fddf = read_dir_of_csv(feedbacks_dir, sep="\t")
    # fix feedback-files issues
    fix_qasrl_id(fddf)
    fddf['noun'] = fddf.apply(lambda r: r.sentence.split()[r.verb_idx], axis=1)
    annotations_with_nmr = fddf[fddf.feedback.astype(str).str.lower().str.contains("nmr")]
    nmr_cases = set(zip(annotations_with_nmr.noun, annotations_with_nmr.verb_form))
    return nmr_cases


def fix_qasrl_id(annot_df) -> NoReturn:
    """
    if the qasrl_id column is in the wrong form 'SentenceId(wikinews:99999:0:0)',
    fix it to 'wikinews:99999:0:0'.
    """
    annot_df['qasrl_id'] = annot_df.qasrl_id.apply(lambda s:s.lstrip('SentenceId(').rstrip(')'))


def set_feedback_column(annot_df: pd.DataFrame, feedbacks: pd.DataFrame) -> NoReturn:
    fix_qasrl_id(feedbacks)
    from qanom.annotations.common import set_key_column
    set_key_column(feedbacks)
    key2feedback = dict(zip(feedbacks.key, feedbacks.feedback))
    annot_df['feedback'] = annot_df.key.apply(key2feedback.get)


"""
Functions for adjusting the "duplicated" annotation to the complete dataset - select one worker triplet, 
and take their annotations as regular (non-duplicated) annotations.  
"""


def prune_duplicated_annot(gen_dupl_df: pd.DataFrame, arb_dupl_df: pd.DataFrame) -> pd.DataFrame:

    def get_annot_of_single_worker(df: pd.DataFrame) -> pd.DataFrame:
        """ For each predicate, get only annotations of one worker."""
        delim = '--'
        df['workers_str'] = df.groupby('key').worker_id.transform(lambda v: delim.join(set(v)))
        df['worker_index'] = df.apply(lambda r: r['workers_str'].split(delim).index(r.worker_id), axis=1)
        return df[df['worker_index'] == 0].drop(['workers_str', 'worker_index'], axis=1)

    sing_arb_df = get_annot_of_single_worker(arb_dupl_df)
    # now find those predicates that are in generation but not in single arbitration -
    # and take one "isVerbal==False" response into final csv
    keys_in_arb = set(sing_arb_df.key)
    gen_df_required = gen_dupl_df[(~gen_dupl_df.key.isin(keys_in_arb)) & (~gen_dupl_df.is_verbal)]
    # key only single row (response) per predicate
    gen_df_required = gen_df_required.drop_duplicates('key')
    combined_df = pd.concat([gen_df_required, sing_arb_df], ignore_index=True, sort=False)
    final_df = convert_to_final_annot(combined_df, get_anonymization(all_worker_ids))
    return final_df


def generate_pruned_dupl_annot() -> NoReturn:
    gen_dupl_fn = "files/annotations/gold_set/generation/corrected_filtered/annot.dupl.wikinews.dev.5.csv"
    arb_dupl_fn = "files/annotations/gold_set/arbitration/arbit.dupl.wikinews.dev.5.csv"
    out_fn = "files/annotations/gold_set/final/annot.final.wikinews.dev.5.csv"
    gen_dupl_df = common.read_annot_csv(gen_dupl_fn)
    arb_dupl_df = common.read_annot_csv(arb_dupl_fn)
    pruned_final_df = prune_duplicated_annot(gen_dupl_df, arb_dupl_df)
    common.save_annot_csv(pruned_final_df, out_fn)


"""
Functions for producing the final annotations - taking the .arbit file and adding the 
predicates with isVerbal==false,false from generation (that haven't been sent to consolidation). 
"""

# all workers participating in our annotation project
all_worker_ids = {'A21LONLNBOB8Q', 'A2A4UAFZ5LW71K',
                  'A3IR7DFEKLLLO', 'A98E8M4QLI9RS',
                  'AJQGWGESKQT4Y', 'A1FS8SBR4SDWYG',
                  'A25AX0DNHKJCQT'}


def generate_final_annotation_files() -> NoReturn:
    """
    Generating the final gold annotations -
    1. taking the .arbit file and adding the predicates
    with isVerbal==false,false from generation (that haven't been sent to consolidation).
    2. Anonymize worker-id
    3. Adjust CSV columns
    """
    gen_dir_path = "files/annotations/gold_set/generation/corrected_filtered"
    arb_dir_path = "files/annotations/gold_set/arbitration"
    dest_path = "files/annotations/gold_set/final"
    arb_name_to_gen_name = lambda name: '.'.join(['annot'] + name.split('.')[1:])
    ann_files = [(os.path.join(arb_dir_path, fn), os.path.join(gen_dir_path, arb_name_to_gen_name(fn)))
                 for fn in os.listdir(arb_dir_path)
                 if fn.endswith(".csv") and arb_name_to_gen_name(fn) in os.listdir(gen_dir_path)]
    from annotations.common import read_annot_csv, save_annot_csv
    # prepare worker anonymization (dataset-wide)
    anonymization: Dict[str, str] = get_anonymization(all_worker_ids)
    for arb_fn, gen_fn in ann_files:
        arb_df = read_annot_csv(arb_fn)
        gen_df = read_annot_csv(gen_fn)
        # combine arb with (false,false) predicates from gen
        combined_df = combine_to_final_annot(arb_df=arb_df, gen_df=gen_df)
        # make internal aesthetic modifications in the DataFrame
        final_df = convert_to_final_annot(combined_df, anonymization)
        # save
        fn = os.path.basename(arb_fn)
        # remove prefix and put new one
        fn = 'annot.final.' + fn.lstrip("arbit.")
        dest_fn = os.path.join(dest_path, fn)
        save_annot_csv(final_df, dest_fn)


def convert_to_final_annot(combined_df: pd.DataFrame, anonymization: Dict[str, str]) -> pd.DataFrame:
    """
    Process annot_df with final modifications for dataset publish -
        1. anonymize worker_id
        2. drop unnecessary columns - 'is_redundant', 'assign_id'
        3. rename columns - 'verb' -> 'noun';  'verb_idx' -> 'target_idx'
    @:argument combined_df: a complete annotation dataframe (that includes arbitrator annotation and the
    "is_verbal=False,False" predicates from generation.
    """
    # anonymize
    anon_df = anonymize_workers(combined_df, anonymization)
    # format data - drop unnecessary columns, rename
    columns_to_drop = {'is_redundant', 'assign_id', 'source_assign_id'} & set(anon_df.columns)
    final_df = anon_df.drop(columns=columns_to_drop)
    utils.rename_column(final_df, 'verb', 'noun')
    utils.rename_column(final_df, 'verb_idx', 'target_idx')
    return final_df


def combine_to_final_annot(arb_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the predicates with isVerbal==false,false from generation to arbitration annotations.
    """
    gen_df_shallow = gen_df.drop_duplicates(['key', 'worker_id'])
    key2ngen = dict(gen_df_shallow.key.value_counts())
    gen_df_shallow['ngen'] = gen_df_shallow.key.apply(key2ngen.get)
    # stay with those predicates with 2 generators
    gen_df_2gen = gen_df_shallow[gen_df_shallow.ngen==2]
    from annotations.analyze import isVerbalSumSeries
    key2isVrbSum = dict(isVerbalSumSeries(gen_df_2gen))
    gen_df_2gen['is_vrb_sum'] = gen_df_2gen.key.apply(key2isVrbSum.get)
    # required predicates - those for which both generators agreed that is_verbal=False
    gen_df_required = gen_df_2gen[gen_df_2gen.is_vrb_sum==0].drop_duplicates('key').drop(['ngen', 'is_vrb_sum'], axis=1)
    final_df = pd.concat([gen_df_required, arb_df], ignore_index=True, sort=False)
    return final_df


def get_anonymization(all_worker_ids: Set[str]) -> Dict[str, str]:
    import random
    random.seed(1000)
    wid_list : List[str] = list(sorted(all_worker_ids))
    random.shuffle(wid_list)
    wid2anon_wid = {wid : "Worker-"+str(wid_list.index(wid)+1)
                    for wid in wid_list}
    return wid2anon_wid


def anonymize_workers(annot_df: pd.DataFrame, worker_id_anonymization: Dict[str, str]) -> pd.DataFrame:
    res = annot_df.copy()
    workers_labels = [label for label in annot_df.columns if 'worker' in label]
    for lbl in workers_labels:
        for wid, new_wid in worker_id_anonymization.items():
            res[lbl] = res[lbl].astype(str).str.replace(wid, new_wid)
    return res


def generate_final_train_annotations() -> NoReturn:
    """
    Generating the final train-set annotations -
    1. Anonymize worker-id
    2. Adjust CSV columns
    """
    orig_train_dir_path = "files/annotations/train_set/filtered"
    dest_path = "files/annotations/train_set/final"
    ann_files = [os.path.join(orig_train_dir_path, fn)
                 for fn in os.listdir(orig_train_dir_path) if fn.endswith('.csv')]
    from annotations.common import read_annot_csv, save_annot_csv
    # prepare worker anonymization (dataset-wide)
    anonymization: Dict[str, str] = get_anonymization(all_worker_ids)
    for gen_fn in ann_files:
        gen_df = read_annot_csv(gen_fn)
        # make internal aesthetic modifications in the DataFrame
        final_df = convert_to_final_annot(gen_df, anonymization)
        # save
        fn = os.path.basename(gen_fn)
        dest_fn = os.path.join(dest_path, fn)
        save_annot_csv(final_df, dest_fn)

