from .base import BasicGraph
from .all_pairs_shortest_path import all_pairs_shortest_paths


def shortest_tree_all_pairs(graph):
    """
       'minimize the longest distance between any pair'
    Note: This algorithm is not shortest path as it jumps
    to a new branch when it has exhausted a branch in the tree.
    :return: path
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    g = all_pairs_shortest_paths(graph)
    assert isinstance(g, dict)

    distance = float("inf")
    best_starting_point = -1
    # create shortest path gantt diagram.
    for start_node in g.keys():
        if start_node in g:
            dist = sum(v for k, v in g[start_node].items())
            if dist < distance:
                best_starting_point = start_node
            # else: skip the node as it's isolated.
    g2 = g[best_starting_point]  # {1: 0, 2: 1, 3: 2, 4: 3}

    inv_g2 = {}
    for k, v in g2.items():
        if v not in inv_g2:
            inv_g2[v] = set()
        inv_g2[v].add(k)

    all_nodes = set(g.keys())
    del g
    path = []
    while all_nodes and inv_g2.keys():
        v_nearest = min(inv_g2.keys())
        for v in inv_g2[v_nearest]:
            all_nodes.remove(v)
            path.append(v)
        del inv_g2[v_nearest]
    return path