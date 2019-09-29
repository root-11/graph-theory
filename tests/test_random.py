from matplotlib import pyplot as plt

from graph import Graph, tsp
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

    links = (sum(range(nodes)) * 2)  # this is a fully connected graph.
    g = random_xy_graph(nodes=nodes, edges=links, x_max=800, y_max=400, seed=42)

    # edges=None creates a fully connected graph
    g2 = random_xy_graph(nodes=nodes, edges=None, x_max=800, y_max=400, seed=42)
    assert len(g.nodes()) == len(g2.nodes())
    assert len(g.edges()) == len(g2.edges())


def test_random_graph_3():

    def plot_tour(tour, style='bo-'):
        "Plot every city and link in the tour, and highlight start city."
        if len(tour) > 1000: plt.figure(figsize=(15, 10))
        start = tour[0:1]
        plot_segment(tour + start, style)
        plot_segment(start, 'rD')  # start city is red Diamond.

    def plot_segment(segment, style='bo-'):
        "Plot every city and link in the segment."
        xs = [X(g.node(c)) for c in segment]
        ys = [Y(g.node(c)) for c in segment]
        plt.plot(xs, ys, style, clip_on=False)
        plt.axis('scaled')
        plt.axis('off')

    def X(node):
        """X coordinate."""
        return node[0]

    def Y(node):
        """Y coordinate."""
        return node[1]

    g = random_xy_graph(500, x_max=800, y_max=400)
    dist, tour = tsp(g)
    plot_tour(tour)
    plt.show()