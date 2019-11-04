from graph import Graph
from tests.test_graph import graph02, graph_cycle_6, graph_cycle_5


def test_subgraph():
    g = graph02()
    g2 = g.subgraph_from_nodes([1, 2, 3, 4])
    d = {1: {2: 1, 4: 1},
         2: {3: 1},
         }
    assert g2.is_subgraph(g)
    for k, v in d.items():
        for k2, d2 in v.items():
            assert g.edge(k, k2) == g2.edge(k, k2)

    g3 = graph02()
    g3.add_edge(3, 100, 7)
    assert not g3.is_subgraph(g2)


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
    g = graph02()
    assert not g.has_cycles()


def test_is_really_cyclic():
    g = Graph(from_list=[(1, 1, 1), (2, 2, 1)])  # two loops onto themselves.
    assert g.has_cycles()


def test_components():
    g = Graph(from_list=[
        (1, 2, 1),  # component 1
        (2, 1, 1),
        (3, 3, 1),  # component 2
        (4, 5, 1),
        (5, 6, 1),  # component 3
        (5, 7, 1),
        (6, 8, 1),
        (7, 8, 1),
        (8, 9, 1),
    ])
    g.add_node(10)  # component 4
    components = g.components()
    assert len(components) == 4
    assert {1, 2} in components
    assert {3} in components
    assert {4, 5, 6, 7, 8, 9} in components
    assert {10} in components


def test_phaselines():
    """
     1 +---> 3 +--> 5 +---> 6          [7]
                    ^       ^
       +------------+       |
       |
     2 +---> 4 +----------> +
    """
    g = Graph(from_list=[
        (1, 3, 1),
        (2, 4, 1),
        (2, 5, 1),
        (3, 5, 1),
        (4, 6, 1),
        (5, 6, 1),
    ])
    g.add_node(7)

    p = g.phase_lines()
    assert set(g.nodes()) == set(p.keys())
    expects = {1: 0, 2: 0, 7: 0, 3: 1, 4: 1, 5: 2, 6: 3}
    assert p == expects, (p, expects)


