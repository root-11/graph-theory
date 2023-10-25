import itertools
from graph.core import Graph


def binary_tree(levels):
    """
    Generates a binary tree with the given number of levels.
    """
    if not isinstance(levels, int):
        raise TypeError(f"Expected int, not {type(levels)}")
    g = Graph()
    for i in range(levels):
        g.add_edge(i, 2 * i + 1)
        g.add_edge(i, 2 * i + 2)
    return g


def grid(length, width, bidirectional=False):
    """
    Generates a grid with the given length and width.
    """
    if not isinstance(length, int):
        raise TypeError(f"Expected int, not {type(length)}")
    if not isinstance(width, int):
        raise TypeError(f"Expected int, not {type(width)}")
    g = Graph()
    node_index = {}
    c = itertools.count(start=1)
    for i in range(length):  # i is the row
        for j in range(width):  # j is the column
            node_index[(i, j)] = next(c)
            if i > 0:
                a, b = node_index[(i, j)], node_index[(i - 1, j)]
                g.add_edge(b, a, bidirectional=bidirectional)
            if j > 0:
                a, b = node_index[(i, j)], node_index[(i, j - 1)]
                g.add_edge(b, a, bidirectional=bidirectional)
    return g
