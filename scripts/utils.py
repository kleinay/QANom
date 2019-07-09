
def dictOfLists(pairs):
    # return a { key : [values given to that key] } for the pair list.
    # e.g. dictOfLists( [(0, "r"), (4, "s"), (0, "e")])  will return {0: ["r", "e"], 4: ["s"]}
    from collections import defaultdict
    r = defaultdict(list)
    for k, v in pairs:
        r[k].append(v)
    return dict(r)
