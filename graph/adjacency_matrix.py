from .base import BasicGraph


def adjacency_matrix(graph):
    """Converts directed graph to an adjacency matrix.
    :param graph:
    :return: dictionary
    The distance from a node to itself is 0 and distance from a node to
    an unconnected node is defined to be infinite. This does not mean that there
    is no path from a node to another via other nodes.
    Example:
        g = Graph(from_dict=
            {1: {2: 3, 3: 8, 5: -4},
             2: {4: 1, 5: 7},
             3: {2: 4},
             4: {1: 2, 3: -5},
             5: {4: 6}})
        adjacency_matrix(g)
        {1: {1: 0, 2: 3, 3: 8, 4: inf, 5: -4},
         2: {1: inf, 2: 0, 3: inf, 4: 1, 5: 7},
         3: {1: inf, 2: 4, 3: 0, 4: inf, 5: inf},
         4: {1: 2, 2: inf, 3: -5, 4: 0, 5: inf},
         5: {1: inf, 2: inf, 3: inf, 4: 6, 5: 0}}
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")

    return {
        v1: {v2: 0 if v1 == v2 else graph.edge(v1, v2, default=float("inf")) for v2 in graph.nodes()}
        for v1 in graph.nodes()
    }
