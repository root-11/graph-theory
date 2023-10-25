import math, itertools
from graph.core import Graph
from graph.generate import binary_tree, grid, random_graph, n_products, nth_product

from tests.test_graph import graph3x3, graph4x4, graph5x5


def test_generate_grid_3x3():
    g = grid(3, 3)
    expected = graph3x3()
    assert g == expected


def test_generate_grid_4x4_bidirectional():
    g = grid(4, 4, bidirectional=True)
    expected = graph4x4()
    assert g == expected


def test_generate_grid_5x5_bidirectional():
    g = grid(5, 5, bidirectional=True)
    expected = graph5x5()
    assert g == expected


def test_generate_binary_tree_3_levels():
    g = binary_tree(3)
    expected = Graph(from_list=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]])
    assert g == expected


def test_nth_p():
    a, b, c, d = [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7]
    for i, comb in enumerate(itertools.product(*[a, b, c, d])):
        if i < 5 or i > 836 or i == 444:
            print(i, comb)
        assert comb == nth_product(i, a, b, c, d)


def test_nth_products():
    a, b, c = [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]
    L = list(n_products(10, a, b))
    assert len(L) == 10
    assert L == [(1, 2), (1, 6), (2, 4), (3, 1), (3, 5), (4, 2), (4, 6), (5, 4), (6, 1), (6, 5)]

    L = list(n_products(10, a, b, c))
    assert len(L) == 10
    # fmt:off
    assert L == [(1, 2, 5), (1, 6, 3), (2, 4, 1), (3, 1, 4), (3, 5, 2), (4, 2, 5), (4, 6, 3), (5, 4, 1), (6, 1, 4), (6, 5, 2)]
    # fmt:on


def test_random_graphs():
    g1 = random_graph(10, 1.7, 1)
    g2 = random_graph(10, 1.7, 1)
    assert g1 == g2
    g3 = random_graph(10, 1.1, 1)
    assert g1 != g3
