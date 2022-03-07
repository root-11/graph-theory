from graph import Graph
from graph.random import random_xy_graph


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


def test_random_graph_4():
    """ check that the string method is correct. """
    g = random_xy_graph(1000, 1000, 1000, 7000)
    assert len(g.edges()) == 7000
    s = str(g)
    assert s == 'Graph(1000 nodes, 7000 edges)', s
