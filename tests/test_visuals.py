from graph.random import random_xy_graph, xy_distance
from graph.search import tsp
from graph.visuals import plot_2d
from graph import Graph
from tests.test_spatial_graph import spiral_graph, fishbone_graph


def test_bad_input():
    g = Graph(from_list=[
        (0, 1, 1),  # 3 edges:
    ])
    try:
        _ = plot_2d(g)
        raise AssertionError
    except ValueError:
        pass


def test_bad_input1():
    g = Graph(from_list=[
        ((1,), (0, 1), 1),
    ])
    try:
        _ = plot_2d(g)
        raise AssertionError
    except ValueError:
        pass


def test_bad_input2():
    g = Graph(from_list=[
        ((1, 2), ('a', 'b'), 1),
    ])
    try:
        _ = plot_2d(g)
        raise AssertionError
    except ValueError:
        pass


def test_bad_input3():
    g = Graph(from_list=[
        ((1, 'b'), ('b', 1), 1)
    ])
    try:
        _ = plot_2d(g)
        raise AssertionError
    except ValueError:
        pass


def test_random_graph_3():
    g = random_xy_graph(200, x_max=800, y_max=400)  # a fully connected graph.
    dist, tour = tsp(g)

    # convert the route to a graph.
    g = Graph()

    a = tour[0]
    for b in tour[1:]:
        g.add_edge(a, b, xy_distance(a, b))
        a = b
    # add the link back to start.
    b = tour[0]
    g.add_edge(a, b, xy_distance(a, b))

    # add a red diamond for the starting point.
    plt = plot_2d(g)
    start = tour[0:1]
    xs, ys = [c[0] for c in start], [c[1] for c in start]
    plt.plot(xs, ys, 'rD', clip_on=False)
    plt.show()


def test_plotting():
    g = spiral_graph()
    g.plot()

    g = fishbone_graph()
    plt = g.plot()
    plt.show()
    plt = g.plot(rotation='yxz')
    plt.show()
    plt = g.plot(maintain_aspect_ratio=True)
    plt.show()

    try:
        _ = g.plot(rotation='x')
        raise AssertionError
    except ValueError:
        pass
    try:
        _ = g.plot(rotation='abc')
        raise AssertionError
    except ValueError:
        pass
