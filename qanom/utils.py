from collections import defaultdict
from typing import Dict, List, Iterable, Union, NoReturn

import pandas as pd


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
