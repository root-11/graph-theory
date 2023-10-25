from graph.core import Graph
from graph.generate import binary_tree, grid

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