#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

pd.set_option('display.max_colwidth',400)

"""
logic required:
* prepare prompts-csv from results-csv (include expert annotation)
* evaluate annotation results 
    * IAA agreement
    * Agreement vs. expert columns
    
"""

full_df = pd.read_csv("nom_identification_60p_results_both_filters_and__withExpert.csv")
df = full_df[["Input.id", "Input.context", "Input.intent", "Input.is_factuality_event", 
              "Answer.is_action_noun_yes.on", "Answer.verbForm", "WorkerId"]]


# In[26]:


def as_prompts_df(results_df):
    """ Prepare prompts-csv from results-csv (include expert annotation) """
    return results_df.rename(mapper=lambda s:s.replace("Input.",""), axis="columns")          [['id', 'context', 'intent', 'noun', 'verb_forms', 'Exp_Nom_orig', u'Exp_Crucial', u'Exp_Nom_Class', u'Exp_Nom_Inclusive',
           u'Exp_Nom_num_args','is_had_auto_verbs', u'in_possible_noms']] \
        .drop_duplicates() \
        .set_index("id")
# prompts_df=as_prompts_df(full_df)
# prompts_df.to_csv("nom_identification_prompts_with_expert.csv")



# In[74]:


# df = pd.read_csv("nom_identification_60p_results_verbal_events_Crowd.csv")        .rename(mapper=lambda s:s.replace("Input.",""), axis="columns")
def compute_iaa(df):
    """ Compute Inter Annotator Agreement - how many prompts were agreed for in the binary classification question (is-nom)"""
    nHits = df.id.nunique()
    gSer=df.groupby('id')['Answer.is_action_noun_yes.on']  # groupBySeries by prompt-id (or otherwise HITId)
    agreedPrompts = (gSer.max() == gSer.min()) # agreedPrompts is a boolean series, per id, is it agreed among all workers
    nAgreed = agreedPrompts.sum()
    iaa = float(nAgreed) / nHits
    print("Agreeing on "+ str(nAgreed) + " out of " + str(nHits) + " ({}%)".format(iaa))
    return iaa

prediction_column = 'Answer.is_action_noun_yes.on'


def compare_to_gold_column(df, gold_column, prediction_column='Answer.is_action_noun_yes.on'):
    agreeSer = (df[prediction_column] == df[gold_column]) # boolean Series (same rows) whether prediction is agreeeing with gold
    accuracy = float(agreeSer.sum()) / agreeSer.size # how many correct predictions 
    gold_nominalizations = df[gold_column].sum()
    print("%s: ( %d \\ %d correct nominalizations); prediction overall accuracy is %.2f" %         (gold_column, gold_nominalizations, agreeSer.size, accuracy))
    predicted_positive = df[prediction_column].sum()
    predicted_negative = (~df[prediction_column]).sum()
    tp = (df[gold_column] & df[prediction_column]).sum()
    fp = ((~df[gold_column]) & df[prediction_column]).sum()
    fn = (df[gold_column] & (~df[prediction_column])).sum()
    print("tp:",tp,"\\",predicted_positive, "\t fp:", fp,"\\",predicted_positive, "\t fn:",fn,"\\",predicted_negative)
    precision = float(tp)/predicted_positive
    recall = float(tp)/gold_nominalizations
    f1 = 2*precision*recall / (precision+recall)
    print("f1:%.2f \t P:%.2f \t R: %.2f" % (f1, precision, recall))
    
def compare_to_columns(df, gold_columns):
    for gold_column in gold_columns:
        compare_to_gold_column(df, gold_column)

def compare_to_exp_nom(df):
    df['Exp_has_args'] = df[u'Exp_Nom_num_args']!=0
    gold_columns = [u'Exp_Nom_Inclusive', 'Exp_Nom_orig', 'Exp_has_args', u'Exp_Crucial']
    compare_to_columns(df, gold_columns)

