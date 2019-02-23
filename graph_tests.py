import time
import random
from . import Graph
from itertools import combinations


def test01():
    """
    Asserts that the shortest_path is correct
    """
    d = {1: {2: 10, 3: 5},
         2: {4: 1, 3: 2},
         3: {2: 3, 4: 9, 5: 2},
         4: {5: 4},
         5: {1: 7, 4: 6}}
    G = Graph()
    G.update_from_dict(d)

    dist, path = G.shortest_path(1, 4)
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
    G = Graph()
    G.update_from_dict(d)
    assert d[3] == G[3]
    assert d[3][4] == G[3][4]


def test_graph_data():
    """triggers error in priorityDictionary
    g and h are two dictionaries with exactly same structure, but some distances vary
    g triggers an error and h doesn't
    """
    g = {1: {2: 1, 3: 9, 4: 4, 5: 13, 6: 20},
         2: {1: 7, 3: 7, 4: 2, 5: 11, 6: 18},
         3: {8: 20, 4: 4, 5: 4, 6: 16, 7: 16},
         4: {8: 15, 3: 4, 5: 9, 6: 11, 7: 21},
         5: {8: 11, 6: 2, 7: 17},
         6: {8: 9, 7: 5},
         7: {8: 3},
         8: {7: 5}}
    h = {1: {2: 1, 3: 9, 4: 4, 5: 11, 6: 17},
         2: {1: 7, 3: 7, 4: 2, 5: 9, 6: 15},
         3: {8: 17, 4: 4, 5: 4, 6: 14, 7: 13},
         4: {8: 12, 3: 4, 5: 9, 6: 9, 7: 18},
         5: {8: 9, 6: 2, 7: 15},
         6: {8: 9, 7: 5},
         7: {8: 3},
         8: {7: 5}}

    G = Graph()
    G.update_from_dict(h)
    dist, path = G.shortest_path(1, 8)
    assert path == [1, 2, 4, 8], path
    G = Graph()
    G.update_from_dict(g)
    dist, path = G.shortest_path(1, 8)
    assert path == [1, 2, 4, 8], path


def test_same():
    p1 = [1, 2, 3, 4, 5]
    p2 = [3, 4, 5, 1, 2]
    g = Graph()
    assert g.same_path(p1, p2)


def test_tsp():
    """
    Assures that the optimization method works.

    The greedy algorithm of the TSP solvers first method
    will build this graph:

    (0)---(1)---(2)---(3)---(4)
       \                     /
        ----------------------------------------
                          /                     \
                        (5)---(6)---(7)---(8)---(9)

    The method 'improve tour' will then reverse segments
    if they improve the overall tour length. This leads to
    this result:

    (0)---(1)---(2)---(3)---(4)--------------\
       \                                      \
        \                                      \
          \                                     \
            \-----------(5)---(6)---(7)---(8)---(9)

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

    def distance(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx ** 2 + dy ** 2) ** (1 / 2)

    # The graph must be fully connected for the TSP to work:
    for a, b in combinations(range(len(xys)), 2):
        d = distance(xys[a], xys[b])
        g.add_link(a, b, distance=d)
        g.add_link(b, a, distance=d)

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

    def distance(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx ** 2 + dy ** 2) ** (1 / 2)

    # The graph must be fully connected for the TSP to work:
    for a, b in combinations(range(len(xys)), 2):
        d = distance(xys[a], xys[b])
        g.add_link(a, b, distance=d)
        g.add_link(b, a, distance=d)

    dist, path = g.solve_tsp()
    expected_tour = [i for i in range(len(xys))]
    expected_length = len(xys)
    assert dist == expected_length, (dist, expected_length)
    g.same_path(path, expected_tour)


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
        g.add_link(a, b, distance=d)
        g.add_link(b, a, distance=d)

    start = time.clock()
    dist, path = g.solve_tsp()
    end = time.clock()
    print("Running tsp on {} points, took {:.3f} seconds".format(points, end - start))
    assert len(path) == points



