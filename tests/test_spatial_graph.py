from math import sin, cos, isclose
from graph import Graph3D


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


def fishbone_graph(levels=5, lengths=10, depths=2):
    """ Creates a multi level fishbone graph.

    :param levels: int: number of levels
    :param lengths: int: number of ribs
    :param depths: int: number of joints on each rib.
    :return: Graph3D
    """
    g = Graph3D()
    prev_level = None
    for level in range(1, levels+1):  # z axis.
        g.add_node((0, 0, level))
        if prev_level is None:
            pass
        else:
            g.add_edge((0, 0, prev_level), (0, 0, level), value=1, bidirectional=True)

        prev_spine = (0, 0, level)  # the lift.
        for step in range(1, lengths+1):  # step along the x axis.
            spine = (step, 0, level)
            g.add_edge(prev_spine, spine, 1, bidirectional=True)

            for side in [-1, 1]:
                rib_1 = spine
                for depth in range(1, depths+1):
                    rib_2 = (step, side * depth, level)
                    g.add_edge(rib_1, rib_2, 1, bidirectional=True)
                    rib_1 = rib_2

            prev_spine = spine
        prev_level = level

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

    g4 = g.copy()
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


def test_shortest_path():
    """ assure that the fishbone graphs entry and exits are connected. """
    g = fishbone_graph()
    entry_point = (-1, 0, 2)
    exit_point = (-1, 0, 1)
    d, p = g.shortest_path(entry_point, exit_point)
    assert d == 3, d


def test_no_nearest_neighbour():
    """ checks that when you're alone, you have no neighbours."""
    g = Graph3D()
    xyz = (1, 1, 1)
    g.add_node(xyz)
    assert g.n_nearest_neighbours(xyz) is None


def test_bfs():
    g = fishbone_graph()
    entry_point = (-1, 0, 2)
    exit_point = (-1, 0, 1)
    d, p = g.breadth_first_search(entry_point, exit_point)
    assert d == 3, d
    assert len(p) == 4, p


def test_dfs():
    g = fishbone_graph()
    entry_point = (-1, 0, 2)
    exit_point = (-1, 0, 1)
    p = g.depth_first_search(entry_point, exit_point)
    assert len(p) == 4, p


def test_distance_from_path():
    g = fishbone_graph()
    entry_point = (-1, 0, 2)
    exit_point = (-1, 0, 1)
    d, p = g.breadth_first_search(entry_point, exit_point)
    assert d == 3
    assert len(p) == 4
    assert g.distance_from_path(p) == 3


def test_maximum_flow():
    g = fishbone_graph()
    entry_point = (-1, 0, 2)
    exit_point = (-1, 0, 1)
    total_flow, flow_graph = g.maximum_flow(entry_point, exit_point)
    assert total_flow == 1, total_flow
    assert len(flow_graph.nodes()) == 4, len(flow_graph.nodes())


def test_subgraph_from_nodes():
    g = fishbone_graph()
    entry_point = (-1, 0, 2)
    exit_point = (-1, 0, 1)
    d, p = g.breadth_first_search(entry_point, exit_point)
    subgraph = g.subgraph_from_nodes(p)
    assert isinstance(subgraph, Graph3D), type(subgraph)
    new_d, new_p = subgraph.breadth_first_search(entry_point, exit_point)
    assert p == new_p, (p, new_p)


def test_has_cycles():
    g = Graph3D()
    a, b, c = (0, 0, 0), (1, 1, 1), (2, 2, 2)
    g.add_edge(a, b, 1)
    g.add_edge(b, c, 1)
    assert not g.has_cycles()
    g.add_edge(c, a, 1)
    assert g.has_cycles()


def test_network_size():
    g = fishbone_graph(levels=3, lengths=3, depths=3)
    try:
        plt = g.plot(rotation='yxz')
        plt.show()
    except ImportError:
        pass
    entry_point = (-1, 0, 2)
    assert g.network_size(entry_point)


def test_has_path():
    g = Graph3D()
    a, b, c = (0, 0, 0), (1, 1, 1), (2, 2, 2)
    g.add_edge(a, b, 1)
    g.add_edge(b, c, 1)
    assert not g.has_path([c, a, b, c])
    g.add_edge(c, a, 1)
    assert g.has_path([c, a, b])
    assert g.has_path([a, b, c])


def test_degree_of_separation():
    g = fishbone_graph()
    entry_point = (-1, 0, 2)
    exit_point = (-1, 0, 1)
    des = g.degree_of_separation(entry_point, exit_point)
    assert des == 3


def test_number_of_components():
    g = fishbone_graph(3, 3, 3)
    cs = g.components()
    assert len(cs) == 1, cs

    g = fishbone_graph()
    cs = g.components()
    assert len(cs) == 1, cs


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
