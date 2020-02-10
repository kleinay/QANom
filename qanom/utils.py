from collections import defaultdict
from typing import Dict, List, Iterable, Union, NoReturn

import pandas as pd


def rename_column(df: pd.DataFrame, orig_label: str, new_label: str):
    df.rename(mapper={orig_label:new_label}, axis=1, inplace=True)

def concatCsvs(csv_fn_list: List[str], output_fn: str) -> NoReturn:
    inp_dfs = [pd.read_csv(fn) for fn in csv_fn_list]
    concatenated_df = pd.concat(inp_dfs)
    concatenated_df.to_csv(output_fn)
    print("output DataFrame shape: ", concatenated_df.shape)


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
        target_dict[newKey] = orig_dict.get(oldKey)
        if inplace: orig_dict.pop(oldKey)
    return target_dict


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
