from functools import lru_cache
from collections import defaultdict
import math
import random
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


def nth_permutation(idx, length, alphabet=None, prefix=()):
    if alphabet is None:
        alphabet = [i for i in range(length)]
    if length == 0:
        return prefix
    else:
        branch_count = math.factorial(length - 1)
        for d in alphabet:
            if d not in prefix:
                if branch_count <= idx:
                    idx -= branch_count
                else:
                    return nth_permutation(idx, length - 1, alphabet, prefix + (d,))


def nth_product(idx, *args):
    """returns the nth product of the given iterables.

    Args:
        idx (int): the index.
        *args: the iterables.
    """
    if not isinstance(idx, int):
        raise TypeError(f"Expected int, not {type(idx)}")
    total = math.prod([len(a) for a in args])
    if idx < 0:
        idx += total
    if index < 0 or index >= total:
        raise IndexError(f"Index {index} out of range")
    
    elements = ()
    for i in range(len(args)):
        offset = math.prod([len(a) for a in args[i:]]) // len(args[i])
        index = idx // offset
        elements += (args[i][index],)
        idx -= index * offset
    return elements


def n_products(*args, n=20):
    """
    Returns the nth product of the given iterables.
    """
    if len(args) == 0:
        return ()
    if any(len(a) == 0 for a in args):
        raise ZeroDivisionError("Cannot generate products of empty iterables")

    n = min(n, math.prod([len(a) for a in args]))
    step = math.prod([len(a) for a in args]) / n

    for ni in range(n):
        ix = int(step * ni + step / 2)
        yield nth_product(ix, *args)
    

def random_graph(size, degree=1.7, seed=1):
    if not isinstance(size, int):
        raise TypeError(f"Expected int, not {type(size)}")
    if not isinstance(degree, float):
        raise TypeError(f"Expected float, not {type(degree)}")
    if not isinstance(seed, int):
        raise TypeError(f"Expected int, not {type(seed)}")

    g = Graph()
    nodes = list(range(size))
    rng = random.Random(seed)
    rng.shuffle(nodes)

    edges = size * degree

    gen_length = math.factorial(nodes) / math.factorial(nodes - 2)
    comb = int(gen_length / edges)

    for i in nodes:
        n = i * comb
        a, b = nth_permutation(n, size, nodes)
        g.add_edge(a, b)
    return g
