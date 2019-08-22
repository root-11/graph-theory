from graph import BasicGraph


def adjacency_matrix(graph):
    """
    :param graph:
    :return: dictionary

    Converts directed graph to an adjacency matrix.
    Note: The distance from a node to itself is 0 and distance from a node to
    an unconnected node is defined to be infinite. This does not mean that there
    is no path from a node to another via other nodes.
        g = {1: {2: 3, 3: 8, 5: -4},
             2: {4: 1, 5: 7},
             3: {2: 4},
             4: {1: 2, 3: -5},
             5: {4: 6}}
        adj(g)
        {1: {1: 0, 2: 3, 3: 8, 4: inf, 5: -4},
         2: {1: inf, 2: 0, 3: inf, 4: 1, 5: 7},
         3: {1: inf, 2: 4, 3: 0, 4: inf, 5: inf},
         4: {1: 2, 2: inf, 3: -5, 4: 0, 5: inf},
         5: {1: inf, 2: inf, 3: inf, 4: 6, 5: 0}}
    """
    assert isinstance(graph, BasicGraph)
    return {v1: {v2: 0 if v1 == v2 else graph.edge(v1, v2, default=float('inf'))
                 for v2 in graph.nodes()}
            for v1 in graph.nodes()}


def all_pairs_shortest_paths(graph):
    """
    Find the cost of the shortest path between every pair of vertices in a
    weighted graph. Uses the Floyd-Warshall algorithm.

    inf = float('inf')
    g = {0: {0: 0,   1: 1,   2: 4},
         1: {0: inf, 1: 0,   2: 2},
         2: {0: inf, 1: inf, 2: 0}}
    fw(g) #
    {0: {0: 0,   1: 1,   2: 3},
    1: {0: inf, 1: 0,   2: 2},
    2: {0: inf, 1: inf, 2: 0}}
    h = {1: {2: 3, 3: 8, 5: -4},
         2: {4: 1, 5: 7},
         3: {2: 4},
         4: {1: 2, 3: -5},
         5: {4: 6}}
    fw(adj(h)) #
        {1: {1: 0, 2: 1, 3: -3, 4: 2, 5: -4},
         2: {1: 3, 2: 0, 3: -4, 4: 1, 5: -1},
         3: {1: 7, 2: 4, 3: 0, 4: 5, 5: 3},
         4: {1: 2, 2: -1, 3: -5, 4: 0, 5: -2},
         5: {1: 8, 2: 5, 3: 1, 4: 6, 5: 0}}
    """
    g = graph.adjacency_matrix()
    assert isinstance(g, dict)
    vertices = g.keys()

    for v2 in vertices:
        g = {v1: {v3: min(g[v1][v3], g[v1][v2] + g[v2][v3])
                  for v3 in vertices}
             for v1 in vertices}
    return g