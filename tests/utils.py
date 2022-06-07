from collections import defaultdict
from graph import Graph


def is_sequence_valid(sequence, graph):
    """ helper to verify that the suggested path actually exists."""

    d = defaultdict(list)
    for item in sequence:
        for k, t in item.items():
            if k not in d:
                d[k].extend(t)
            elif d[k][-1] == t[0]:
                d[k].append(t[-1])
            else:
                raise ValueError

    return all(graph.has_path(p) for k, p in d.items())


def is_matching(a, b):
    """ Helper to check that the moves in A are the same as in B."""
    g1 = Graph()
    for d in a:
        for k,v in d.items():
            g1.add_edge(*v, bidirectional=True)
    g2 = Graph()
    for d in b:
        for k,v in d.items():
            g2.add_edge(*v, bidirectional=True)
    return g1 == g2