from graph import Graph
from tests.test_graph import graph02, graph03


def test_adjacency_matrix():
    g = graph02()
    am = g.adjacency_matrix()
    g2 = Graph(from_dict=am)
    assert g.is_subgraph(g2)
    assert g2._max_edge_value == float('inf') != g._max_edge_value
    assert not g2.is_subgraph(g)


def test_all_pairs_shortest_path():
    g = graph03()
    d = g.all_pairs_shortest_paths()
    g2 = Graph(from_dict=d)
    for n1 in g.nodes():
        for n2 in g.nodes():
            if n1 == n2:
                continue
            d, path = g.shortest_path(n1, n2)
            d2 = g2.edge(n1, n2)
            assert d == d2

    g2.add_node(100)
    d = g2.all_pairs_shortest_paths()
    # should trigger print of isolated node.