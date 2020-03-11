import random
import time
from itertools import combinations

from graph import Graph
from tests.test_graph import graph02, graph03, graph04, graph05, graph_cycle_5


def test_shortest_path01():
    g = graph03()
    dist_g, path_g = g.shortest_path(1, 8)
    assert path_g == [1, 2, 4, 8], path_g

    h = graph04()
    distH, pathH = g.shortest_path(1, 8)
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


def test_distance():
    g = graph02()
    p = [1, 2, 3, 6, 9]
    assert g.distance_from_path(p) == 4

    assert float('inf') == g.distance_from_path([1, 2, 3, 900])  # 900 doesn't exist.


def test_bfs():
    g = graph03()
    d, path = g.breadth_first_search(1, 7)
    assert d == 2, d
    assert path == [1, 3, 7], path

    d, path = g.breadth_first_search(1, 900)  # 900 doesn't exit.
    assert d == float('inf')
    assert path == []


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
    expected = [[1, 2, 3, 6], [1, 2, 5, 6], [1, 4, 5, 6]]
    assert all(i in expected for i in paths) and all(i in paths for i in expected)


def test_path_permutations03():
    g = graph02()
    paths = g.all_paths(1, 9)
    assert len(paths) == 6
    expected_result = [[1, 2, 3, 6, 9],
                       [1, 2, 5, 6, 9],
                       [1, 2, 5, 8, 9],
                       [1, 4, 5, 6, 9],
                       [1, 4, 5, 8, 9],
                       [1, 4, 7, 8, 9]]
    assert all(i in expected_result for i in paths) and all(i in paths for i in expected_result)


def test_path_permutations04():
    g = Graph(from_list=[(1, 2, 1), (1, 3, 1), (2, 4, 1), (3, 4, 1)])
    paths = g.all_paths(1, 4)
    expected = [[1, 2, 4], [1, 3, 4]]
    assert all(i in expected for i in paths) and all(i in paths for i in expected)


def test_path_permutations_pmk():
    """
    [1] --> [2] --> [3] --> [4] --> [5]
             ^               |
             |               v
    [10] --> +<---- [6] <----+
     ^               |
     |               v
    [9] <-- [8] <-- [7]

    """
    links = [(1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (6, 2), (6, 7), (7, 8), (8, 9), (9, 10), (10, 2)]
    g = Graph(from_list=[(a, b, 1) for a, b in links])
    paths = g.all_paths(start=1, end=5)
    expected = [[1, 2, 3, 4, 5],
                [1, 2, 3, 4, 6, 2, 3, 4, 5],
                [1, 2, 3, 4, 6, 7, 8, 9, 10, 2, 3, 4, 5]]
    assert all(i in expected for i in paths) and all(i in paths for i in expected)


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


def test_dfs_on_cycle():
    edges = [
        (1, 2, 1),
        (2, 3, 1),
        (3, 2, 1),
    ]
    for i in range(3, 10, 1):
        edge = (i, i + 1, 1)
        edges.append(edge)
    g = Graph(from_list=edges)
    path = g.depth_first_search(start=1, end=9)
    assert path is not None, path


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


def test_degree_of_separation():
    g = graph05()
    assert g.degree_of_separation(0, 10) == 3
