from graph.visuals import plot_2d
from graph import Graph


def test_bad_input():
    g = Graph(from_list=[
        (0, 1, 1),  # 3 edges:
    ])
    try:
        _ = plot_2d(g)
        raise AssertionError
    except (ValueError, ImportError):
        pass


def test_bad_input1():
    g = Graph(from_list=[
        ((1,), (0, 1), 1),
    ])
    try:
        _ = plot_2d(g)
        raise AssertionError
    except (ValueError, ImportError):
        pass


def test_bad_input2():
    g = Graph(from_list=[
        ((1, 2), ('a', 'b'), 1),
    ])
    try:
        _ = plot_2d(g)
        raise AssertionError
    except (ValueError, ImportError):
        pass


def test_bad_input3():
    g = Graph(from_list=[
        ((1, 'b'), ('b', 1), 1)
    ])
    try:
        _ = plot_2d(g)
        raise AssertionError
    except (ValueError, ImportError):
        pass

