import json
from collections import defaultdict
from typing import Dict, List, Iterable, Union, NoReturn, Tuple, Any

import pandas as pd


def almost_same(str1: str, str2: str) -> bool:
    from fuzzywuzzy import fuzz
    return fuzz.ratio(str1, str2)>90

def split_str_twice(string: str, del1: str, del2: str) -> Tuple[str, str, str]:
    """ Return a 3-element tuple by applying two splits on a str using two different delimiters.
        only regard the first occurrence of each delimiter.
      E.g.  split_str_twice("an_email@gmail.com", "_", "@") -> ("an", "email", "gmail.com")
      """
    first, second_and_third = string.split(del1,1)
    second, third = second_and_third.split(del2,1)
    return first, second, third

def is_empty_string_series(series: pd.Series) -> pd.Series:
    """ Return boolean Series capturing whether `series` is empty string or NA """
    series = series.fillna('')
    isEmpty = series==''
    return isEmpty

def count_empty_string(series: pd.Series) -> int:
    """ Return number of empty strings or NA in a Series """
    return is_empty_string_series(series).sum()

def rename_column(df: pd.DataFrame, orig_label: str, new_label: str, inplace=True):
    return df.rename(mapper={orig_label:new_label}, axis=1, inplace=inplace)

def concatCsvs(csv_fn_list: List[str], output_fn: str, columns: List[str] = None) -> NoReturn:
    inp_dfs = [pd.read_csv(fn) for fn in csv_fn_list]
    concatenated_df = pd.concat(inp_dfs, ignore_index=True, sort=False)
    # `columns` can determine subset (and order) of output columns
    if columns:
        concatenated_df = concatenated_df[columns]
    concatenated_df.to_csv(output_fn, index=False, encoding="utf-8")
    print(f"exported DataFrame with shape {concatenated_df.shape} to {output_fn}")
    return concatenated_df


def df_to_dict(df: pd.DataFrame, key_column: str, value_column: str) -> Dict[Any, Any]:
    """ Return a dict compiled from two columns in a DataFrame. """
    return dict(zip(df[key_column], df[value_column]))

def df_first_row(df: pd.DataFrame) -> pd.Series:
    return list(df.iterrows())[0][1]

def asRelative(distribution : Union[List, Dict]):
    # get a list\dict of numbers (a distribution), return the relative distribution (element/sum)
    if 'values' in dir(distribution):
        # a dict type
        sm = float(sum(distribution.values()))
        return {k: v / sm for k, v in distribution.items()}
    else:
        # a list type
        sm = float(sum(distribution))
        return [e / sm for e in distribution]

def replaceKeys(orig_dict, oldKeys2NewKeys, inplace=True):
    """ replace keys with new keys using oldKeys2NewKeys mapping. """
    target_dict = orig_dict if inplace else {}
    for oldKey, newKey in oldKeys2NewKeys.items():
        if oldKey in orig_dict:
            target_dict[newKey] = orig_dict.get(oldKey)
            if inplace: orig_dict.pop(oldKey)
    return target_dict

def removeKeys(orig_dict: Dict[Any, Any], keys: List[Any], inplace=True) -> Dict[Any, Any]:
    """ remove these keys from the orig_dict. """
    if not inplace:
        orig_dict = orig_dict.copy()
    for key in keys:
        orig_dict.pop(key)
    return orig_dict

def dictOfLists(pairs):
    # return a { key : [values given to that key] } for the pair list.
    # e.g. dictOfLists( [(0, "r"), (4, "s"), (0, "e")])  will return {0: ["r", "e"], 4: ["s"]}
    r = defaultdict(list)
    for k, v in pairs:
        r[k].append(v)
    return dict(r)


def majority(lst: Iterable[bool], whenTie=True) -> bool:
    lst = list(lst)
    s = sum(lst)
    if s == len(lst)/2.0:
        return whenTie
    else:
        return s > len(lst)/2.0

# jsonl (json-lines) util

class jsonl:
    @classmethod
    def dumps(cls, iterable: Iterable) -> str:
        return "\n".join(json.dumps(obj) for obj in iterable)

    @classmethod
    def dump(cls, iterable: Iterable, fp):
        s = jsonl.dumps(iterable)
        fp.write(s)

    @classmethod
    def loads(cls, lines_str: str) -> List[Any]:
        l = [json.loads(line) for line in lines_str.splitlines()]
        return l

    @classmethod
    def load(cls, fp) -> List[Any]:
        return jsonl.loads(fp.read())



# list, dict and sets utils
def is_iterable(e):
    return '__iter__' in dir(e)


def flatten(lst, recursively=False):
    """ Flatten a list.
    if recursively=True, flattens all levels of nesting, until reaching non-iterable items
    (strings are considered non-iterable to that matter.)
    :returns a flatten list (a non-nested list)
    """
    if not is_iterable(lst):
        return lst
    out = []
    for element in lst:
        if is_iterable(element):
            if recursively:
                out.extend(flatten(element))
            else:
                out.extend(element)
        else:
            out.append(element)
    return out


def is_nested(lst):
    return any(is_iterable(e) for e in lst)


def power_set(lst, as_list=True):
    """ set as_list to false in order to yield the power-set """
    import itertools
    pwset_chain = itertools.chain.from_iterable(itertools.combinations(lst, r)
                                                for r in range(len(lst) + 1))
    if as_list:
        return list(pwset_chain)
    else:
        return pwset_chain


def static_variables(**kwargs):
    """ A decorator for creating static local variables for a function.
    Usage Example:

    @static_variables(counter=0, large_list=load_large_list())
    def foo():
        foo.counter += 1    # now 'counter' and 'large_list' are properties of the method,
                            #  and are initialized only once, in the decorator line.
        print "Counter is %d" % foo.counter
        print foo.large_list[foo.counter]

    """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def lazy_static_variables(**kwargs):
    """ A decorator for creating static local variables for a function with lazy initialization,
    i.e., variables are defined and initialized only
    Usage Example:

    @static_variables(counter=0, large_list=load_large_list())
    def foo():
        foo.counter += 1    # now 'counter' and 'large_list' are properties of the method,
                            #  and are initialized only once, in the decorator line.
        print "Counter is %d" % foo.counter
        print foo.large_list[foo.counter]

    """
    def decorate(func):
        print("decorator")
        for k in kwargs:
            setattr(func, k, kwargs[k]())
        return func
    return decorate
