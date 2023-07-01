from .base import BasicGraph
from .bfs import breadth_first_search


def degree_of_separation(graph, n1, n2):
    """Calculates the degree of separation between 2 nodes."""
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    if n1 not in graph:
        raise ValueError("n1 not in graph.")
    if n2 not in graph:
        raise ValueError("n2 not in graph.")

    assert n1 in graph.nodes()
    p = breadth_first_search(graph, n1, n2)
    return len(p) - 1