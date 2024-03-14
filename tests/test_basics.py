from graph import Graph
from tests.test_graph import graph3x3, graph01, graph05, graph_cycle_6, graph_cycle_5


def test_to_from_dict():
    d = {1: {2: 10, 3: 5},
         2: {4: 1, 3: 2},
         3: {2: 3, 4: 9, 5: 2},
         4: {5: 4},
         5: {1: 7, 4: 6},
         6: {}}
    g = Graph()
    g.from_dict(d)
    d2 = g.to_dict()
    assert d == d2


def test_setitem():
    g = Graph()
    try:
        g[1][2] = 3
        raise ValueError("Assignment is not permitted use g.add_edge instead.")
    except ValueError:
        pass
    g.add_node(1)
    try:
        g[1][2] = 3
        raise ValueError
    except ValueError:
        pass
    g.add_edge(1, 2, 3)
    assert g.edges() == [(1, 2, 3)]
    link_1 = g.edge(1, 2)
    assert link_1 == 3
    link_1 = g.edge(1, 2)
    assert link_1 == 3
    link_1 = 4  # attempt setattr.
    assert g.edge(1, 2) != 4  # the edge is not an object.
    g.add_edge(1, 2, 4)
    assert g.edges() == [(1, 2, 4)]

    g = Graph()
    try:
        g[1] = {2: 3}
        raise ValueError
    except ValueError:
        pass


def test_add_node_attr():
    g = graph3x3()
    g.add_node(1, "this")
    assert set(g.nodes()) == set(range(1, 10))
    node_1 = g.node(1)
    assert node_1 == "this"

    d = {"This": 1, "That": 2}
    g.add_node(1, obj=d)
    assert g.node(1) == d

    rm = 5
    g.del_node(rm)
    for n1, n2, d in g.edges():
        assert n1 != rm and n2 != rm
    g.del_node(rm)  # try again for a node that doesn't exist.


def test_add_edge_attr():
    g = Graph()
    try:
        g.add_edge(1, 2, {'a': 1, 'b': 2})
        raise Exception("Assignment of non-values is not supported.")
    except ValueError:
        pass


class MyCustomHashableNode():
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        """ note that without __eq__ this wont work.
        https://stackoverflow.com/questions/9010222/why-can-a-python-dict-have-multiple-keys-with-the-same-hash?noredirect=1&lq=1
        """
        return hash(self) == hash(other)


def test_node_types():
    for test in [
        [1, 2, 1, 3],
        ['A', 'B', 'A', 'C'],
        [MyCustomHashableNode(i) for i in ['A', 'B', 'A', 'C']],
    ]:
        a,b,c,d = test
        g = Graph()
        g.add_edge(a,b,10)
        g.add_edge(c,d,10)
        assert len(g.nodes()) == 3


def test_to_list():
    g1 = graph01()
    g1.add_node(44)
    g2 = Graph(from_list=g1.to_list())
    assert g1.edges() == g2.edges()
    assert g1.nodes() == g2.nodes()


def test_bidirectional_link():
    g = Graph()
    g.add_edge(node1=1, node2=2, value=4, bidirectional=True)
    assert g.edge(1, 2) == g.edge(2, 1)


def test_edges_with_node():
    g = graph3x3()
    edges = g.edges(from_node=5)
    assert set(edges) == {(5, 6, 1), (5, 8, 1)}
    assert g.edge(5, 6) == 1
    assert g.edge(5, 600) is None  # 600 doesn't exist.


def test_nodes_from_node():
    g = graph3x3()
    nodes = g.nodes(from_node=1)
    assert set(nodes) == {2, 4}
    nodes = g.nodes(to_node=9)
    assert set(nodes) == {6, 8}
    nodes = g.nodes()
    assert set(nodes) == set(range(1, 10))

    try:
        _ = g.nodes(in_degree=-1)
        assert False
    except ValueError:
        assert True

    nodes = g.nodes(in_degree=0)
    assert set(nodes) == {1}
    nodes = g.nodes(in_degree=1)
    assert set(nodes) == {2, 3, 4, 7}
    nodes = g.nodes(in_degree=2)
    assert set(nodes) == {5, 6, 8, 9}
    nodes = g.nodes(in_degree=3)
    assert nodes == []

    try:
        _ = g.nodes(out_degree=-1)
        assert False
    except ValueError:
        assert True

    nodes = g.nodes(out_degree=0)
    assert set(nodes) == {9}

    g.add_node(44)
    assert set(g.nodes(out_degree=0)) == {9, 44}

    nodes = g.nodes(out_degree=1)
    assert set(nodes) == {3, 6, 7, 8}
    nodes = g.nodes(out_degree=2)
    assert set(nodes) == {1, 2, 4, 5}
    nodes = g.nodes(out_degree=3)
    assert nodes == []

    try:
        _ = g.nodes(in_degree=1, out_degree=1)
        assert False
    except ValueError:
        assert True


def test01():
    """
    Asserts that the shortest_path is correct
    """
    g = graph01()
    dist, path = g.shortest_path(1, 4)
    assert [1, 3, 2, 4] == path, path
    assert 9 == dist, dist


def test02():
    """
    Assert that the dict loader works.
    """
    d = {1: {2: 10, 3: 5},
         2: {4: 1, 3: 2},
         3: {2: 3, 4: 9, 5: 2},
         4: {5: 4},
         5: {1: 7, 4: 6}}
    g = Graph(from_dict=d)
    assert 3 in g
    assert d[3][4] == g.edge(3, 4)


def test03():
    g = graph3x3()
    all_edges = g.edges()
    edges = g.edges(path=[1, 2, 3, 6, 9])
    for edge in edges:
        assert edge in all_edges, edge


def test_subgraph():
    g = graph3x3()
    g2 = g.subgraph_from_nodes([1, 2, 3, 4])
    d = {1: {2: 1, 4: 1},
         2: {3: 1},
         }
    assert g2.is_subgraph(g)
    for k, v in d.items():
        for k2, d2 in v.items():
            assert g.edge(k, k2) == g2.edge(k, k2)

    g3 = graph3x3()
    g3.add_edge(3, 100, 7)
    assert not g3.is_subgraph(g2)


def test_in_and_out_degree():
    g = graph3x3()

    in_degree = {1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 1, 8: 2, 9: 2}
    out_degree = {1: 2, 2: 2, 3: 1, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 0}
    for node in g.nodes():
        assert g.in_degree(node) == in_degree[node]
        assert g.out_degree(node) == out_degree[node]

def test_equals():
    g1 = Graph(from_list=[(1,2),(2,3)])
    g2 = g1.copy()
    assert g1 == g2

    g1.add_edge(3,1)
    g1.del_edge(3,1)
    assert g1 == g2

    assert g1.edges() == g2.edges()
    assert g1.nodes() == g2.nodes()

def test_copy_equals():
    g1 = Graph(from_list=[(1,2),(2,3)])
    g1.add_edge(3,1)
    g1.del_edge(3,1)
    g2 = g1.copy()
    assert g1 == g2


def test_copy():
    g = graph05()
    g2 = g.copy()
    assert set(g.edges()) == set(g2.edges())
    assert g == g2, "testing == operator failed"
    g2.add_node(1, "this")

    g3 = Graph(from_list=g.to_list())
    assert g2 != g3
    g2.add_node(1)
    assert g2 == g3


def test_errors():
    g = graph05()
    try:
        len(g)
        raise AssertionError
    except ValueError:
        assert True

    try:
        g.edges(from_node=1, to_node=1)
        raise AssertionError
    except ValueError:
        assert True

    try:
        g.edges(path=[1])
        raise AssertionError
    except ValueError:
        assert True

    try:
        g.edges(path="this")
        raise AssertionError
    except ValueError:
        assert True

    e = g.edges(from_node=77)
    assert e == []


def test_delitem():
    g = graph05()

    try:
        g.__delitem__(key=1)
        assert False
    except ValueError:
        assert True

    g.del_edge(node1=0, node2=1)
    g.del_edge(0, 1)  # idempotent.

    v = g.edge(0, 1)
    if v is not None:
        raise ValueError
    v = g.edge(0, 1, default=44)
    if v != 44:
        raise ValueError


def test_is_partite():
    g = graph_cycle_6()
    bol, partitions = g.is_partite(n=2)
    assert bol is True

    g = graph_cycle_5()
    bol, part = g.is_partite(n=2)
    assert bol is False
    bol, part = g.is_partite(n=5)
    assert bol is True
    assert len(part) == 5


def test_is_cyclic():
    g = graph_cycle_5()
    assert g.has_cycles()


def test_is_not_cyclic():
    g = graph3x3()
    assert not g.has_cycles()


def test_is_really_cyclic():
    g = Graph(from_list=[(1, 1, 1), (2, 2, 1)])  # two loops onto themselves.
    assert g.has_cycles()


def test_no_edge_connected():
    g = Graph()
    g.add_node(4)
    g.add_node(5)
    assert g.is_connected(4, 5) is False


def test_edge_connected():
    g = Graph()
    g.add_edge(4, 5)
    assert g.is_connected(4, 5) is True


def test_edge_not_connected():
    g = Graph()
    g.add_edge(3, 4)
    g.add_edge(5, 4)
    assert g.is_connected(3, 5) is False


def test_del_edge():
    the_sort = []
    g = Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    zero_in = g.nodes(in_degree=0)
    while zero_in:
        for n in zero_in:
            the_sort.append(n)
            g.del_node(n)
        zero_in = g.nodes(in_degree=0)
    assert len(g.nodes()) == 0
    assert len(g._nodes) == 0
    assert len(g._edges) == 0
    assert len(g._reverse_edges) == 0
    assert len(g._in_degree) == 0
    assert len(g._out_degree) == 0
