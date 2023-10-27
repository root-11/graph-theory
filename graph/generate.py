import math
import random
import itertools
from functools import reduce
from operator import mul
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


def nth_product(index, *args):
    """Equivalent to ``list(product(*args))[index]``.

    The products of *args* can be ordered lexicographically.
    :func:`nth_product` computes the product at sort position *index* without
    computing the previous products.

        >>> nth_product(8, range(2), range(2), range(2), range(2))
        (1, 0, 0, 0)

    ``IndexError`` will be raised if the given *index* is invalid.
    """
    pools = list(map(tuple, reversed(args)))
    ns = list(map(len, pools))

    c = reduce(mul, ns)

    if index < 0:
        index += c

    if not 0 <= index < c:
        raise IndexError

    result = []
    for pool, n in zip(pools, ns):
        result.append(pool[index % n])
        index //= n

    return tuple(reversed(result))


def nth_products(n, *args):
    """
    Returns the n evenly spread combinations using 
    nth product of the given iterables.

    Args:
        n (int): the number of products to generate.
        *args: the iterables.
    """
    if len(args) == 0:
        return ()
    if any(len(a) == 0 for a in args):
        raise ZeroDivisionError("Cannot generate products of empty iterables")

    n = min(n, int(math.prod([len(a) for a in args])))
    step = math.prod([len(a) for a in args]) / n

    for ni in range(n):
        ix = int(step * ni + step / 2)
        yield nth_product(ix, *args)


def random_graph(size, degree=1.7, seed=1):
    """Generates a graph with randomized edges

    Args:
        size (int): number of nodes
        degree (float, optional): Average degree of connectivity. Defaults to 1.7.
        seed (int, optional): Random seed. Defaults to 1.

    Returns:
        Graph: the generated graph
    """
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

    edges = int(size * degree)

    L = nth_products(edges, nodes, nodes)
    for a, b in L:
        g.add_edge(a, b)
    return g
