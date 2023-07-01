from .base import BasicGraph


def topological_sort(graph, key=None):
    """Return a generator of nodes in topologically sorted order.
    :param graph: Graph
    :param key: optional function for sortation.
    :return: Generator
    Topological sort (ordering) is a linear ordering of vertices.
    https://en.wikipedia.org/wiki/Topological_sorting
    Note: The algorithm does not check for loops before initiating
    the sortation, but raise AttributeError at the first conflict.
    This saves O(m+n) runtime.
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")

    if key is None:

        def key(x):
            return x

    g2 = graph.copy()

    zero_in_degree = sorted(g2.nodes(in_degree=0), key=key)

    while zero_in_degree:
        for task in zero_in_degree:
            yield task  # <--- do something.

            g2.del_node(task)

        zero_in_degree = sorted(g2.nodes(in_degree=0), key=key)

    if g2.nodes():
        raise AttributeError(f"Graph is not acyclic: Loop found: {g2.nodes()}")