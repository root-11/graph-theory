import random
import time
from itertools import combinations, permutations

from graph import Graph
from tests.test_graph import graph01, graph3x3, graph03, graph04, graph05, graph4x4
from tests.test_graph import london_underground,munich_firebrigade_centre

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
    g = graph3x3()
    d, p = g.shortest_path(start=9, end=1)  # there is no path.
    assert d == float('inf')
    assert p == []


def test_distance():
    g = graph3x3()
    p = [1, 2, 3, 6, 9]
    assert g.distance_from_path(p) == 4

    assert float('inf') == g.distance_from_path([1, 2, 3, 900])  # 900 doesn't exist.


def test_bfs():
    g = graph03()
    path = g.breadth_first_search(1, 7)
    assert path == [1, 3, 7], path

    path = g.breadth_first_search(1, 900)  # 900 doesn't exit.
    assert path == []

def test_bfw():
    g = graph03()
    bfw = g.breadth_first_walk(1)
    walk = [n for n in bfw]
    assert walk == [1, 2, 3, 4, 5, 6, 8, 7], walk

    bfw = g.breadth_first_walk(1,5)
    walk = [n for n in bfw]
    assert walk == [1, 2, 3, 4, 5]


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


def test_all_paths_no_path():
    """
    [1] --> [2]    [3] --> [4]
    """
    g = Graph(from_list=[(1,2,1), (3,4,1)])
    paths = g.all_paths(1, 4)
    assert paths == []


def test_all_paths_start_is_end():
    g = graph3x3()
    try:
        g.all_paths(2, 2)
        raise AssertionError("a value error should have been raised.")
    except ValueError:
        pass


def test_all_paths01():
    g = graph3x3()
    paths = g.all_paths(1, 3)
    assert len(paths) == 1, paths
    assert paths[0] == [1, 2, 3]


def test_all_paths02():
    g = graph3x3()
    paths = g.all_paths(1, 6)
    assert len(paths) == 3
    expected = [[1, 2, 3, 6], [1, 2, 5, 6], [1, 4, 5, 6]]
    assert all(i in expected for i in paths) and all(i in paths for i in expected)


def test_all_paths03():
    g = graph3x3()
    paths = g.all_paths(1, 9)
    assert len(paths) == 6
    expected_result = [[1, 2, 3, 6, 9],
                       [1, 2, 5, 6, 9],
                       [1, 2, 5, 8, 9],
                       [1, 4, 5, 6, 9],
                       [1, 4, 5, 8, 9],
                       [1, 4, 7, 8, 9]]
    assert all(i in expected_result for i in paths) and all(i in paths for i in expected_result)


def test_all_paths04():
    g = Graph(from_list=[(1, 2, 1), (1, 3, 1), (2, 4, 1), (3, 4, 1)])
    paths = g.all_paths(1, 4)
    expected = [[1, 2, 4], [1, 3, 4]]
    assert all(i in expected for i in paths) and all(i in paths for i in expected)


def test_all_paths05():
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


def test_all_paths06():
    links = [(1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (6, 2), (6, 7), (7, 8), (8, 9), (9, 10), (10, 2)]
    g = Graph(from_list=[(a, b, 1) for a, b in links])
    for comb in permutations(list(range(1, 11)), 2):
        start, end = comb
        _ = g.all_paths(start, end)
    assert True, "All permutations of start and end passed."


def test_dfs():
    links = [
        (1, 2, 0),
        (1, 3, 0),
        (2, 3, 0),
        (2, 4, 0),
        (3, 4, 0)
    ]
    g = Graph(from_list=links)
    try:
        g.depth_first_search(0,2)
        assert False, "node 0 is not in g"
    except ValueError:
        pass

    try:
        g.depth_first_search(1,99)
        assert False, "node 99 is not in g"
    except ValueError:
        pass

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


def test_depth_scan_01():
    links = [
        (1, 2, 0),
        (1, 3, 0),
        (3, 5, 0),
        (2, 4, 0),
        (5, 6, 0),
    ]
    g = Graph(from_list=links)

    def visit_node(node) -> bool:
        return node != 2

    visited = g.depth_scan(1, visit_node)
    assert len(visited) == 5
    assert 1 in visited
    assert 2 in visited
    assert 3 in visited
    assert 5 in visited
    assert 6 in visited
    assert 4 not in visited


def test_depth_scan_02():
    """ criteria not callable"""
    g = graph01()

    criteria = 41  # not callable
    try:
        g.depth_scan(1, criteria)
        assert False, "criteria must be a callable, so this is not possible"
    except TypeError:
        assert True


def test_depth_scan_03():
    """ start not in graph """
    g = graph01()
    start_that_doesnt_exist = max(g.nodes()) + 1

    def criteria(n):
        return False

    try:
        g.depth_scan(start_that_doesnt_exist, criteria)
        assert False, "start isn't in g, so reaching this code isn't possible."
    except ValueError:
        assert True


def test_depth_scan_04():
    """ criteria negative on start"""
    g = graph01()

    def criteria(n):
        return False

    empty_set = set()
    assert g.depth_scan(1, criteria) == empty_set


def test_depth_scan_05():
    g = graph01()

    def criteria(n):
        return n < 5

    result = g.depth_scan(1, criteria)
    assert max(result) == 5, result


def test_degree_of_separation():
    g = graph05()
    assert g.degree_of_separation(0, 10) == 3


def test_loop():
    g = graph4x4()
    p = Graph.loop(g, 1, 16)
    assert p == [1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5, 1]


def test_avoids():
    g = graph4x4()
    p = Graph.avoids(g, 1, 16, (3, 7, 11, 10))
    assert p == [1, 5, 9, 13, 14, 15, 16], p


def test_incomparable_path_searching():
    """
    incomparable type A -> incomparable type A -> incomparable type B
    |
    v
    incomparable type C
    """
    g = Graph()
    g.add_edge(("A", "6"), ("B", "7"))
    g.add_edge(("A", "6"), 6)
    g.add_edge(("B", "7"), "B")

    p = g.all_paths(("A", "6"), "B")
    assert p == [[("A", "6"), ("B", "7"), "B"]]

    p = g.depth_first_search(("A", "6"), "B")
    assert p == [("A", "6"), ("B", "7"), "B"]

    p = g.breadth_first_search(("A", "6"), "B")
    assert p == [("A", "6"), ("B", "7"), "B"]

    p = g.shortest_path(("A", "6"), "B")
    assert p == (2, [("A", "6"), ("B", "7"), "B"])


def test_cached_graph():
    g = Graph(from_list=[(s, e, d + (s / 100)) for s, e, d in graph4x4().edges()])
    g2 = g.copy()
    a, b = 1, 16
    d2, p2 = g2.shortest_path(a, b, memoize=True)
    d1, p1 = g.shortest_path(a, b)
    assert d1 == d2, (d1, d2)
    assert p1 == p2, (p1, p2)


def test_cached_graph2():
    g = london_underground()
    seds = list(g.edges())
    for s, e, d in seds:
        g.add_edge(s, e, d + s / len(seds) ** 2)  # adding minor variances so that no paths are the same length.

    r1 = g.shortest_path(74, 89, memoize=True)
    r2 = g.shortest_path(74, 89, memoize=True)
    assert r1 == r2, "cache call should be the same as the previous"
    r3 = g.shortest_path(99, 89, memoize=True)  # this is a cache call as p(99,89) is in p(74,89)
    assert r3 == (14.003052109588591, [74, 99, 236, 229, 273, 107, 192, 277, 89])

    a1, b1 = 10, 89
    for a, b in combinations(g.nodes(), 2):
        if a == a1 and b == b1:
            d1, p1 = g.shortest_path(a, b)
            d2, p2 = g.shortest_path(a, b, memoize=True)
            assert d1 == d2
            assert p1 == p2
            break
        else:
            g.shortest_path(a, b, memoize=True)


def test_incremental_search():
    g = munich_firebrigade_centre()

    seds = list(g.edges())
    for s, e, d in seds:
        g.add_edge(s, e, d + s / len(seds)**2)  # adding minor variances so that no paths are the same length.

    t_repeated, t_memoized, cnt = 0.0, 0.0, 0

    for a, b in combinations(g.nodes(), 2):
        start = time.process_time()
        d1, p1 = g.shortest_path(a, b)
        end = time.process_time()
        t_repeated += end - start

        start = time.process_time()
        d2, p2 = g.shortest_path(a, b, memoize=True)
        end = time.process_time()
        t_memoized += end - start
        assert d1 == d2, (d1, d2)
        assert p1 == p2, (p1, p2)

        cnt += 1

        if cnt > 200:
            break

    pct = round(100 * t_memoized / t_repeated)
    print("repeated searches", t_repeated, "secs."
          "\nmemoized searches:", t_memoized, "secs."
          "\ntime using memoising: ", pct, "%",
          flush=True)
    assert t_repeated > t_memoized


