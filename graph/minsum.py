from .base import BasicGraph


def minsum(graph):
    """finds the mode(s) that have the smallest sum of distance to all other nodes."""
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    adj_mat = graph.all_pairs_shortest_paths()
    for n in adj_mat:
        adj_mat[n] = sum(adj_mat[n].values())
    smallest = min(adj_mat.values())
    return [k for k, v in adj_mat.items() if v == smallest]