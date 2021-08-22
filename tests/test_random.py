from graph import Graph, tsp_greedy
from graph.random import random_xy_graph, xy_distance
from graph.visuals import plot_2d, visuals_enabled


def graph07():
    nodes = 10
    links = 30
    g = random_xy_graph(nodes=nodes, edges=links, x_max=800, y_max=400, seed=42)
    assert isinstance(g, Graph)
    assert len(g.nodes()) == nodes
    assert len(g.edges()) == links
    return g


def test_random_graph():
    g = graph07()
    assert isinstance(g, Graph)


def test_random_graph_2():
    nodes = 10000
    links = 1

    err1 = err2 = ""
    try:
        g = random_xy_graph(nodes=nodes, edges=links, x_max=80, y_max=40, seed=42)
    except ValueError as e:
        err1 = str(e)
        pass
    nodes = 10
    links = (sum(range(nodes)) * 2) + 1
    try:
        g = random_xy_graph(nodes=nodes, edges=links, x_max=800, y_max=400, seed=42)
    except ValueError as e:
        err2 = str(e)
        pass
    assert str(err1) != str(err2)

    links = 1
    g = random_xy_graph(nodes=nodes, edges=links, x_max=7, y_max=2, seed=42)
    # the xy_space above is so small that the generator must switch from random
    # mode, to search mode.

    links = nodes * nodes   # this is a fully connected graph.
    g = random_xy_graph(nodes=nodes, edges=links, x_max=800, y_max=400, seed=42)

    # edges=None creates a fully connected graph
    g2 = random_xy_graph(nodes=nodes, edges=None, x_max=800, y_max=400, seed=42)
    assert len(g.nodes()) == len(g2.nodes())
    assert len(g.edges()) == len(g2.edges())


def test_random_graph_3():
    g = random_xy_graph(200, x_max=800, y_max=400)  # a fully connected graph.
    dist, tour = tsp_greedy(g)

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
    if not visuals_enabled:
        return
    plt = plot_2d(g)
    start = tour[0:1]
    xs, ys = [c[0] for c in start], [c[1] for c in start]
    plt.plot(xs, ys, 'rD', clip_on=False)
    plt.show()


def test_random_graph_4():
    """ check that the string method is correct. """
    g = random_xy_graph(1000, 1000, 1000, 7000)
    assert len(g.edges()) == 7000
    s = str(g)
    assert s == 'Graph(1000 nodes, 7000 edges)', s
