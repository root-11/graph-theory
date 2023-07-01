from .base import BasicGraph


def all_pairs_shortest_paths(graph):
    """Find the cost of the shortest path between every pair of vertices in a
    weighted graph. Uses the Floyd-Warshall algorithm.
    Example:
        inf = float('inf')
        g = Graph(from_dict=(
            {0: {0: 0,   1: 1,   2: 4},
             1: {0: inf, 1: 0,   2: 2},
             2: {0: inf, 1: inf, 2: 0}})
        fw(g)
        {0: {0: 0,   1: 1,   2: 3},
        1: {0: inf, 1: 0,   2: 2},
        2: {0: inf, 1: inf, 2: 0}}
        h = {1: {2: 3, 3: 8, 5: -4},
             2: {4: 1, 5: 7},
             3: {2: 4},
             4: {1: 2, 3: -5},
             5: {4: 6}}
        fw(adj(h)) #
            {1: {1: 0, 2:  1, 3: -3, 4: 2, 5: -4},
             2: {1: 3, 2:  0, 3: -4, 4: 1, 5: -1},
             3: {1: 7, 2:  4, 3:  0, 4: 5, 5:  3},
             4: {1: 2, 2: -1, 3: -5, 4: 0, 5: -2},
             5: {1: 8, 2:  5, 3:  1, 4: 6, 5:  0}}
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected BasicGraph, Graph or Graph3D, not {type(graph)}")

    g = graph.adjacency_matrix()
    assert isinstance(g, dict), "previous function should have returned a dict."
    vertices = g.keys()

    for v2 in vertices:
        g = {v1: {v3: min(g[v1][v3], g[v1][v2] + g[v2][v3]) for v3 in vertices} for v1 in vertices}
    return g
