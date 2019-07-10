
def dictOfLists(pairs):
    # return a { key : [values given to that key] } for the pair list.
    # e.g. dictOfLists( [(0, "r"), (4, "s"), (0, "e")])  will return {0: ["r", "e"], 4: ["s"]}
    from collections import defaultdict
    r = defaultdict(list)
    for k, v in pairs:
        r[k].append(v)
    return dict(r)


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
