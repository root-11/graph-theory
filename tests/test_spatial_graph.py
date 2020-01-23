from math import sin, cos
from graph.spatial_graph import Graph3D


def test_basics():
    xyz = [(sin(i / 150), cos(i / 150), i / 150) for i in range(0, 1500, 20)]
    g = Graph3D()
    for t in xyz:
        g.add_node(t)

    for n1 in g.nodes():
        nn = g.n_nearest_neighbours(n1)[0]
        distance = g.distance(n1, nn)
        g.add_edge(n1, nn, distance)

    g.plot()

    L = g.to_list()
    g2 = Graph3D(from_list=L)

    d = g.to_dict()
    g3 = Graph3D(from_dict=d)

    g4 = g.__copy__()


def test_no_nearest_neighbour():
    g = Graph3D()
    xyz = (1, 1, 1)
    g.add_node(xyz)
    assert g.n_nearest_neighbours(xyz) is None  # itself!


def test_bad_config():
    g = Graph3D()
    try:
        g.distance((1, 2), (3, 4))
        raise AssertionError
    except ValueError:
        pass

    try:
        g.add_edge(1, 2)
        raise AssertionError
    except TypeError:
        pass

    try:
        g.add_edge((1,), (2,))
        raise AssertionError
    except ValueError:
        pass

    try:
        g.add_edge((1, 2, 3), (2,))
        raise AssertionError
    except ValueError:
        pass

    try:
        g.add_edge((1, 2, 3), 2)
        raise AssertionError
    except TypeError:
        pass

    try:
        g.add_edge((1.0, 1.0, 1.0), (1, 1.0, "1"))
        raise AssertionError
    except TypeError:
        pass

    try:
        g.add_node(1)
        raise AssertionError
    except TypeError:
        pass

    try:
        g.add_node((1,))
        raise AssertionError
    except ValueError:
        pass

    try:
        g.n_nearest_neighbours(1)
        raise AssertionError
    except TypeError:
        pass

    try:
        g.n_nearest_neighbours((1,))
        raise AssertionError
    except ValueError:
        pass

    try:
        g.n_nearest_neighbours((1, 2, 3), n='abc')
        raise AssertionError
    except TypeError:
        pass

    try:
        g.n_nearest_neighbours((1, 2, 3), n=-1)
        raise AssertionError
    except ValueError:
        pass
