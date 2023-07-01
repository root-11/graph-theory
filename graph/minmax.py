from .base import BasicGraph


def minmax(graph):
    """finds the node(s) with shortest distance to all other nodes."""
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    adj_mat = graph.all_pairs_shortest_paths()
    for n in adj_mat:
        adj_mat[n] = max(adj_mat[n].values())
    smallest = min(adj_mat.values())
    return [k for k, v in adj_mat.items() if v == smallest]