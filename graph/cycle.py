from .base import BasicGraph
from .topological_sort import topological_sort


def cycle(graph, start, mid, end=None):
    """Returns a loop passing through a defined mid-point and returning via a different set of nodes to the outward
    journey. If end is None we return to the start position."""
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    if start not in graph:
        raise ValueError("start not in graph.")
    if mid not in graph:
        raise ValueError("mid not in graph.")
    if end is not None and end not in graph:
        raise ValueError("end not in graph.")

    _, p = graph.shortest_path(start, mid)
    g2 = graph.copy()
    if end is not None:
        for n in p[:-1]:
            g2.del_node(n)
        _, p2 = g2.shortest_path(mid, end)
    else:
        for n in p[1:-1]:
            g2.del_node(n)
        _, p2 = g2.shortest_path(mid, start)
    lp = p + p2[1:]
    return lp


def has_cycles(graph):
    """Checks if graph has a cycle
    :param graph: instance of class Graph.
    :return: bool
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")

    for n1, n2, _ in graph.edges():
        if n1 == n2:  # detect nodes that point to themselves
            return True
    try:
        _ = list(topological_sort(graph))  # tries to create a DAG.
        return False
    except AttributeError:
        return True