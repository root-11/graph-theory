from math import sin, cos, isclose
from graph.spatial_graph import Graph3D


def spiral_graph():
    xyz = [(sin(i / 150), cos(i / 150), i / 150) for i in range(0, 1500, 30)]
    g = Graph3D()
    for t in xyz:
        g.add_node(t)

    xyz.sort(key=lambda x: x[2])  # sorted by z axis which is linear.

    n1 = xyz[0]
    for n2 in xyz[1:]:
        distance = g.distance(n1, n2)
        g.add_edge(n1, n2, distance)
        n1 = n2
    return g


def fishbone_graph():
    g = Graph3D()
    j1 = None
    for j2 in range(5):  # z axis.
        g.add_node((0, 0, j2))
        if j1 is None:
            pass
        else:
            g.add_edge((0, 0, j1), (0, 0, j2), value=1, bidirectional=True)

        s1, l1, r1 = None, None, None  # (s)pine, (l)eft side, (r)ight side
        for i in range(10):  # step along the x axis.
            s2, l2, r2 = (i, 0, j2), (i, 1, j2), (i, -1, j2)
            g.add_node(s2)  # spine
            g.add_node(l2)  # left rib
            g.add_node(r2)  # right rib
            if s1 is None:
                pass
            else:
                g.add_edge(s1, s2, 1)
            g.add_edge(s2, l2, 1, bidirectional=True)
            g.add_edge(s2, r2, 1, bidirectional=True)

            s1, l1, r1 = s2, l2, r2
        j1 = j2

    g.add_node((-1, 0, 2))  # entry point
    g.add_edge((-1, 0, 2), (0, 0, 2), 1, bidirectional=True)
    g.add_node((-1, 0, 1))  # exit point
    g.add_edge((-1, 0, 1), (0, 0, 1), 1, bidirectional=True)
    return g


def test_basics():
    g = Graph3D()
    a, b, c = (0, 0, 0), (1, 1, 1), (2, 2, 2)
    g.add_node(a)
    g.add_node(b)
    g.add_node(c)
    assert g.n_nearest_neighbours(a)[0] == b
    assert g.n_nearest_neighbours(c)[0] == b

    L = g.to_list()
    g2 = Graph3D(from_list=L)
    assert g2.nodes() == g.nodes()
    assert g2.edges() == g.edges()

    d = g.to_dict()
    g3 = Graph3D(from_dict=d)
    assert g3.nodes() == g.nodes()
    assert g3.edges() == g.edges()

    g4 = g.__copy__()
    assert g4.edges() == g.edges()


def test_path_finding():
    xyz = [(sin(i / 150), cos(i / 150), i / 150) for i in range(0, 1500, 30)]
    g = Graph3D()
    for t in xyz:
        g.add_node(t)

    xyz.sort(key=lambda x: x[2])

    total_distance = 0.0
    n1 = xyz[0]
    for n2 in xyz[1:]:
        distance = g.distance(n1, n2)
        total_distance += distance
        g.add_edge(n1, n2, distance)
        n1 = n2

    d, p = g.shortest_path(xyz[0], xyz[-1])
    assert isclose(d, 13.847754085278877), d
    assert p == xyz, [(idx, a, b) for idx, (a, b) in enumerate(zip(p, xyz)) if a != b]


def test_plotting():
    g = spiral_graph()
    g.plot()

    g = fishbone_graph()
    g.plot()
    g.plot(rotation='yxz')
    g.plot(maintain_aspect_ratio=True)

    try:
        g.plot(rotation='x')
    except ValueError:
        pass
    try:
        g.plot(rotation='abc')
    except ValueError:
        pass


def test_shortest_path():
    g = fishbone_graph()
    entry_point = (-1, 0, 2)
    exit_point = (-1, 0, 1)
    d, p = g.shortest_path(entry_point, exit_point)
    assert d == 3


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
