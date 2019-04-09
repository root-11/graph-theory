import time
import random
from graph import *
from itertools import combinations


def graph01():
    """
    :return: Graph.
    """
    d = {1: {2: 10, 3: 5},
         2: {4: 1, 3: 2},
         3: {2: 3, 4: 9, 5: 2},
         4: {5: 4},
         5: {1: 7, 4: 6}}
    return Graph(from_dict=d)


def graph02():
    """
    1 -> 2 -> 3
    |    |    |
    v    v    v
    4 -> 5 -> 6
    |    |    |
    v    v    v
    7 -> 8 -> 9

    :return: :return:
    """
    d = {1: {2: 1, 4: 1},
         2: {3: 1, 5: 1},
         3: {6: 1},
         4: {5: 1, 7: 1},
         5: {6: 1, 8: 1},
         6: {9: 1},
         7: {8: 1},
         8: {9: 1}
         }
    return Graph(from_dict=d)


def graph03():
    d = {1: {2: 1, 3: 9, 4: 4, 5: 13, 6: 20},
         2: {1: 7, 3: 7, 4: 2, 5: 11, 6: 18},
         3: {8: 20, 4: 4, 5: 4, 6: 16, 7: 16},
         4: {8: 15, 3: 4, 5: 9, 6: 11, 7: 21},
         5: {8: 11, 6: 2, 7: 17},
         6: {8: 9, 7: 5},
         7: {8: 3},
         8: {7: 5}}
    return Graph(from_dict=d)


def graph04():
    d = {1: {2: 1, 3: 9, 4: 4, 5: 11, 6: 17},
         2: {1: 7, 3: 7, 4: 2, 5: 9, 6: 15},
         3: {8: 17, 4: 4, 5: 4, 6: 14, 7: 13},
         4: {8: 12, 3: 4, 5: 9, 6: 9, 7: 18},
         5: {8: 9, 6: 2, 7: 15},
         6: {8: 9, 7: 5},
         7: {8: 3},
         8: {7: 5}}
    return Graph(from_dict=d)


def graph05():
    """
    0 ---- 1 ---- 5
     +      +---- 6 ---- 7
      +            +     |
       +            +---- 8
        +
         +- 2 ---- 3 ---- 9
             +      +     |
              4      +---10
    """
    links = [
        (0, 1, 1),
        (0, 2, 1),
        (1, 5, 1),
        (1, 6, 1),
        (2, 3, 1),
        (2, 4, 1),
        (3, 9, 1),
        (3, 10, 1),
        (9, 10, 1),
        (6, 7, 1),
        (6, 8, 1),
        (7, 8, 1),
        (0, 1, 1),
        (0, 1, 1),
        (0, 1, 1),
    ]
    return Graph(from_list=links)


def graph_cycle_6():
    """
    cycle of 6 nodes
    """
    links = [
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (4, 5, 1),
        (5, 6, 1),
        (6, 1, 1),
    ]
    links.extend([(n2, n1, d) for n1, n2, d in links])
    return Graph(from_list=links)


def graph_cycle_5():
    """ cycle of 5 nodes """
    links = [
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (4, 5, 1),
        (5, 1, 1),
    ]
    links.extend([(n2, n1, d) for n1, n2, d in links])
    return Graph(from_list=links)


def test_to_from_dict():
    d = {1: {2: 10, 3: 5},
         2: {4: 1, 3: 2},
         3: {2: 3, 4: 9, 5: 2},
         4: {5: 4},
         5: {1: 7, 4: 6},
         6: {}}
    g = Graph()
    g.from_dict(d)
    d2 = g.to_dict()
    assert d == d2


def test_setitem():
    g = Graph()
    try:
        g[1][2] = 3
        raise ValueError("Assignment is not permitted use g.add_edge instead.")
    except ValueError:
        pass
    g.add_node(1)
    try:
        g[1][2] = 3
        raise ValueError
    except ValueError:
        pass
    g.add_edge(1, 2, 3)
    assert g.edges() == [(1, 2, 3)]
    link_1 = g.edge(1, 2)
    assert link_1 == 3
    link_1 = g.edge(1, 2)
    assert link_1 == 3
    link_1 = 4  # attempt setattr.
    assert g.edge(1, 2) != 4  # the edge is not an object.
    g.add_edge(1, 2, 4)
    assert g.edges() == [(1, 2, 4)]

    g = Graph()
    try:
        g[1] = {2: 3}
        raise ValueError
    except ValueError:
        pass


def test_add_node_attr():
    g = graph02()
    g.add_node(1, "this")
    assert set(g.nodes()) == set(range(1, 10))
    node_1 = g.node(1)
    assert node_1 == "this"

    d = {"This": 1, "That": 2}
    g.add_node(1, obj=d)
    assert g.node(1) == d

    rm = 5
    g.del_node(rm)
    for n1, n2, d in g.edges():
        assert n1 != rm and n2 != rm
    g.del_node(rm)  # try again for a node that doesn't exist.


def test_add_edge_attr():
    g = Graph()
    try:
        g.add_edge(1, 2, {'a': 1, 'b': 2})
        raise Exception("Assignment of non-values is not supported.")
    except ValueError:
        pass


def test_to_list():
    g1 = graph01()
    g1.add_node(44)
    g2 = Graph(from_list=g1.to_list())
    assert g1.edges() == g2.edges()
    assert g1.nodes() == g2.nodes()


def test_bidirectional_link():
    g = Graph()
    g.add_edge(node1=1, node2=2, value=4, bidirectional=True)
    assert g.edge(1, 2) == g.edge(2, 1)


def test_edges_with_node():
    g = graph02()
    edges = g.edges(from_node=5)
    assert set(edges) == {(5, 6, 1), (5, 8, 1)}
    assert g.edge(5, 6) == 1
    assert g.edge(5, 600) is None  # 600 doesn't exist.


def test_nodes_from_node():
    g = graph02()
    nodes = g.nodes(from_node=1)
    assert set(nodes) == {2, 4}
    nodes = g.nodes(to_node=9)
    assert set(nodes) == {6, 8}
    nodes = g.nodes()
    assert set(nodes) == set(range(1, 10))

    try:
        _ = g.nodes(in_degree=-1)
        assert False
    except ValueError:
        assert True

    nodes = g.nodes(in_degree=0)
    assert set(nodes) == {1}
    nodes = g.nodes(in_degree=1)
    assert set(nodes) == {2, 3, 4, 7}
    nodes = g.nodes(in_degree=2)
    assert set(nodes) == {5, 6, 8, 9}
    nodes = g.nodes(in_degree=3)
    assert nodes == []

    try:
        _ = g.nodes(out_degree=-1)
        assert False
    except ValueError:
        assert True

    nodes = g.nodes(out_degree=0)
    assert set(nodes) == {9}
    nodes = g.nodes(out_degree=1)
    assert set(nodes) == {3, 6, 7, 8}
    nodes = g.nodes(out_degree=2)
    assert set(nodes) == {1, 2, 4, 5}
    nodes = g.nodes(out_degree=3)
    assert nodes == []

    try:
        _ = g.nodes(in_degree=1, out_degree=1)
        assert False
    except ValueError:
        assert True


def test01():
    """
    Asserts that the shortest_path is correct
    """
    g = graph01()
    dist, path = g.shortest_path(1, 4)
    assert [1, 3, 2, 4] == path, path
    assert 9 == dist, dist


def test02():
    """
    Assert that the dict loader works.
    """
    d = {1: {2: 10, 3: 5},
         2: {4: 1, 3: 2},
         3: {2: 3, 4: 9, 5: 2},
         4: {5: 4},
         5: {1: 7, 4: 6}}
    g = Graph(from_dict=d)
    assert 3 in g
    assert d[3][4] == g.edge(3, 4)


def test03():
    g = graph02()
    all_edges = g.edges()
    edges = g.edges(path=[1, 2, 3, 6, 9])
    for edge in edges:
        assert edge in all_edges, edge


def test_shortest_path01():
    g = graph03()
    dist_g, path_g = g.shortest_path(1, 8)
    assert path_g == [1, 2, 4, 8], path_g

    h = graph04()
    distH, pathH = h.shortest_path(1, 8)
    assert pathH == [1, 2, 4, 8], pathH
    pathH = pathH[2:] + pathH[:2]
    assert path_g != pathH, (path_g, pathH)

    assert g.same_path(path_g, pathH)
    assert h.same_path(path_g, pathH)

    reverseG = list(reversed(path_g))
    assert not g.same_path(path_g, reverseG)

    assert g.has_path(path_g)


def test_tsp():
    """
    Assures that the optimization method works.

    The greedy algorithm of the TSP solvers first method
    will build this graph:

    (0)---(1)---(2)---(3)---(4)
       +                     /
        ----------------------------------------
                          /                     +
                        (5)---(6)---(7)---(8)---(9)

    The method 'improve tour' will then reverse segments
    if they improve the overall tour length. This leads to
    this result:

    (0)---(1)---(2)---(3)---(4)--------------+
       +                                      +
        +                                      +
          +                                     +
            +-----------(5)---(6)---(7)---(8)---(9)

    which is a fraction shorter.
    """
    g = Graph()
    xys = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7)
    ]

    def _distance(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx ** 2 + dy ** 2) ** (1 / 2)

    # The graph must be fully connected for the TSP to work:
    for a, b in combinations(range(len(xys)), 2):
        d = _distance(xys[a], xys[b])
        g.add_edge(a, b, value=d)
        g.add_edge(b, a, value=d)

    dist, path = g.solve_tsp()
    expected_tour = [0, 1, 2, 3, 4, 9, 8, 7, 6, 5]
    expected_length = 14.32455532033676
    assert dist == expected_length, (dist, expected_length)
    assert g.same_path(path, expected_tour)


def test_tsp_perfect_problem():
    """
    The tour is:

    (0,5)--(1,5)--(2,5)--(3,5)--(4,5)--(5,5)
      |                                  |
    (0,4)                              (5,4)
      |                                  |
    (0,3)                              (5,3)
      |                                  |
    (0,2)                              (5,2)
      |                                  |
    (0,1)--(1,1)--(2,1)--(3,1)--(4,1)--(5,1)

    And it must be 18 long.
    """
    g = Graph()
    xys = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 5), (2, 5), (3, 5), (4, 5), (5, 5),
        (5, 4), (5, 3), (5, 2),
        (5, 1), (4, 1), (3, 1), (2, 1), (1, 1),
    ]

    def _distance(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx ** 2 + dy ** 2) ** (1 / 2)

    # The graph must be fully connected for the TSP to work:
    for a, b in combinations(range(len(xys)), 2):
        d = _distance(xys[a], xys[b])
        g.add_edge(a, b, value=d)
        g.add_edge(b, a, value=d)

    dist, path = g.solve_tsp()
    expected_tour = [i for i in range(len(xys))]
    expected_length = len(xys)
    assert dist == expected_length, (dist, expected_length)
    assert g.same_path(path, expected_tour)


def test_tsp_larger_problem():
    random.seed(44)
    points = 200

    xys = set()
    while len(xys) != points:
        xys.add((random.randint(0, 600), random.randint(0, 800)))
    xys = [n for n in xys]

    g = Graph()
    for a, b in combinations(xys, 2):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        d = (dx ** 2 + dy ** 2) ** (1 / 2)
        g.add_edge(a, b, value=d)
        g.add_edge(b, a, value=d)

    start = time.process_time()
    dist, path = g.solve_tsp()
    end = time.process_time()
    print("Running tsp on {} points, took {:.3f} seconds".format(points, end - start))
    assert len(path) == points


def test_shortest_path_fail():
    g = graph02()
    d, p = g.shortest_path(start=9, end=1)  # there is no path.
    assert d == float('inf')
    assert p == []


def test_subgraph():
    g = graph02()
    g2 = g.subgraph_from_nodes([1, 2, 3, 4])
    d = {1: {2: 1, 4: 1},
         2: {3: 1},
         }
    assert g2.is_subgraph(g)
    for k, v in d.items():
        for k2, d2 in v.items():
            assert g.edge(k, k2) == g2.edge(k, k2)

    g3 = graph02()
    g3.add_edge(3, 100, 7)
    assert not g3.is_subgraph(g2)


def test_distance():
    G = graph02()
    p = [1, 2, 3, 6, 9]
    assert G.distance_from_path(p) == len(p) - 1

    assert float('inf') == G.distance_from_path([1, 2, 3, 900])  # 900 doesn't exist.


def test_bfs():
    g = graph03()
    d, path = g.breadth_first_search(1, 7)
    assert d == 2, d
    assert path == [1, 3, 7], path

    d, path = g.breadth_first_search(1, 900)  # 900 doesn't exit.
    assert d == float('inf')
    assert path == []


def test_adjacency_matrix():
    g = graph02()
    am = g.adjacency_matrix()
    g2 = Graph(from_dict=am)
    assert g.is_subgraph(g2)
    assert g2._max_edge_value == float('inf') != g._max_edge_value
    assert not g2.is_subgraph(g)


def test_all_pairs_shortest_path():
    g = graph03()
    d = g.all_pairs_shortest_paths()
    g2 = Graph(from_dict=d)
    for n1 in g.nodes():
        for n2 in g.nodes():
            if n1 == n2:
                continue
            d, path = g.shortest_path(n1, n2)
            d2 = g2.edge(n1, n2)
            assert d == d2

    g2.add_node(100)
    d = g2.all_pairs_shortest_paths()
    # should trigger print of isolated node.


def test_shortest_tree_all_pairs01():
    g = Graph()
    links = [
        (1, 2, 1),
        (1, 3, 1),
        (2, 3, 1)
    ]
    for L in links:
        g.add_edge(*L)

    p = g.shortest_tree_all_pairs()
    assert p == [1, 2, 3]


def test_shortest_tree_all_pairs02():
    links = [
        (1, 2, 1),
        (1, 3, 2),
        (2, 3, 3)
    ]
    g = Graph(from_list=links)

    for L in links:
        g.add_edge(*L)

    p = g.shortest_tree_all_pairs()
    assert p == [1, 2, 3]


def test_path_permutations01():
    g = graph02()
    paths = g.all_paths(1, 3)
    assert len(paths) == 1, paths
    assert paths[0] == [1, 2, 3]


def test_path_permutations02():
    g = graph02()
    paths = g.all_paths(1, 6)
    assert len(paths) == 3
    assert paths == [[1, 2, 3, 6], [1, 2, 5, 6], [1, 4, 5, 6]]


def test_path_permutations03():
    g = graph02()
    paths = g.all_paths(1, 9)
    assert len(paths) == 6
    assert paths == [[1, 2, 3, 6, 9],
                     [1, 2, 5, 6, 9],
                     [1, 2, 5, 8, 9],
                     [1, 4, 5, 6, 9],
                     [1, 4, 5, 8, 9],
                     [1, 4, 7, 8, 9]], paths


def test_maximum_flow():
    """ [2] ----- [5]
       /    +   /  | +
    [1]      [4]   |  [7]
       +    /   +  | /
        [3] ----- [6]
    """
    links = [
        (1, 2, 18),
        (1, 3, 10),
        (2, 4, 7),
        (2, 5, 6),
        (3, 4, 2),
        (3, 6, 8),
        (4, 5, 10),
        (4, 6, 10),
        (5, 6, 16),
        (5, 7, 9),
        (6, 7, 18)
    ]
    g = Graph(from_list=links)

    flow, g2 = g.maximum_flow(1, 7)
    assert flow == 23, flow


def test_maximum_flow01():
    links = [
        (1, 2, 1)
    ]
    g = Graph(from_list=links)
    flow, g2 = g.maximum_flow(start=1, end=2)
    assert flow == 1, flow


def test_maximum_flow02():
    links = [
        (1, 2, 10),
        (2, 3, 1),  # bottleneck.
        (3, 4, 10)
    ]
    g = Graph(from_list=links)
    flow, g2 = g.maximum_flow(start=1, end=4)
    assert flow == 1, flow


def test_maximum_flow03():
    links = [
        (1, 2, 10),
        (1, 3, 10),
        (2, 4, 1),  # bottleneck 1
        (3, 5, 1),  # bottleneck 2
        (4, 6, 10),
        (5, 6, 10)
    ]
    g = Graph(from_list=links)
    flow, g2 = g.maximum_flow(start=1, end=6)
    assert flow == 2, flow


def test_maximum_flow04():
    links = [
        (1, 2, 10),
        (1, 3, 10),
        (2, 4, 1),  # bottleneck 1
        (2, 5, 1),  # bottleneck 2
        (3, 5, 1),  # bottleneck 3
        (3, 4, 1),  # bottleneck 4
        (4, 6, 10),
        (5, 6, 10)
    ]
    g = Graph(from_list=links)
    flow, g2 = g.maximum_flow(start=1, end=6)
    assert flow == 4, flow


def test_maximum_flow05():
    links = [
        (1, 2, 10),
        (1, 3, 1),
        (2, 3, 1)
    ]
    g = Graph(from_list=links)
    flow, g2 = g.maximum_flow(start=1, end=3)
    assert flow == 2, flow


def test_dfs():
    links = [
        (1, 2, 0),
        (1, 3, 0),
        (2, 3, 0),
        (2, 4, 0),
        (3, 4, 0)
    ]
    g = Graph(from_list=links)
    path = g.depth_first_search(1, 4)
    assert g.has_path(path)

    path = g.depth_first_search(4, 1)
    assert path is None, path


def test_dfs_02():
    links = [
        (1, 2, 0),
        (1, 3, 0),
        (3, 5, 0),
        (2, 4, 0),
        (5, 6, 0),
    ]
    g = Graph(from_list=links)
    path = g.depth_first_search(1, 4)
    assert path == [1, 2, 4]
    assert g.has_path(path)


def test_dfs_03():
    g = graph05()
    path = g.depth_first_search(0, 10)
    assert path == [0, 2, 3, 9, 10]
    assert g.has_path(path)


def test_copy():
    g = graph05()
    g2 = g.__copy__()
    assert set(g.edges()) == set(g2.edges())


def test_delitem():
    g = graph05()

    try:
        g.__delitem__(key=1)
        assert False
    except ValueError:
        assert True

    g.del_edge(node1=0, node2=1)

    v = g.edge(0, 1)
    if v is not None:
        raise ValueError
    v = g.edge(0, 1, default=44)
    if v != 44:
        raise ValueError


def test_is_partite():
    g = graph_cycle_6()
    bol, partitions = g.is_partite(n=2)
    assert bol is True

    g = graph_cycle_5()
    bol, part = g.is_partite(n=2)
    assert bol is False
    bol, part = g.is_partite(n=5)
    assert bol is True
    assert len(part) == 5


def test_is_cyclic():
    g = graph_cycle_5()
    assert g.has_cycles()


def test_is_not_cyclic():
    g = graph02()
    assert not g.has_cycles()


def test_is_really_cyclic():
    g = Graph(from_list=[(1, 1, 1), (2, 2, 1)])  # two loops onto themselves.
    assert g.has_cycles()
