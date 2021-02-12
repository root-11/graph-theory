from graph import Graph, minimum_cost_flow_using_successive_shortest_path


def test_maximum_flow():
    """ [2] ----- [5]
       /    +   /  | +
    [1]      [4]   |  [7]
       +    /   +  | /
        [3] ----- [6]
    """
    edges = [
        (1, 2, 18),
        (1, 3, 10),
        (2, 4, 7),
        (2, 5, 6),
        (3, 4, 2),
        (3, 6, 8),
        (4, 5, 10),
        (4, 6, 10),
        (5, 6, 16),
        (5, 7, 9),
        (6, 7, 18)
    ]
    g = Graph(from_list=edges)

    flow, g2 = g.maximum_flow(1, 7)
    assert flow == 23, flow



def test_min_cut():
    """ [2] ----- [5]
       /    +   /  | +
    [1]      [4]   |  [7]
       +    /   +  | /
        [3] ----- [6]
    """
    edges = [
        (1, 2, 18),
        (1, 3, 18),  # different from test_maximum_flow
        (2, 4, 7),
        (2, 5, 6),
        (3, 4, 2),
        (3, 6, 8),
        (4, 5, 10),
        (4, 6, 10),
        (5, 6, 16),
        (5, 7, 9),
        (6, 7, 18)
    ]
    g = Graph(from_list=edges)

    max_flow_min_cut = g.maximum_flow_min_cut(1,7)
    assert set(max_flow_min_cut) == {(2, 5), (2, 4), (3, 4), (3, 6)}


def test_maximum_flow01():
    edges = [
        (1, 2, 1)
    ]
    g = Graph(from_list=edges)
    flow, g2 = g.maximum_flow(start=1, end=2)
    assert flow == 1, flow


def test_maximum_flow02():
    edges = [
        (1, 2, 10),
        (2, 3, 1),  # bottleneck.
        (3, 4, 10)
    ]
    g = Graph(from_list=edges)
    flow, g2 = g.maximum_flow(start=1, end=4)
    assert flow == 1, flow


def test_maximum_flow03():
    edges = [
        (1, 2, 10),
        (1, 3, 10),
        (2, 4, 1),  # bottleneck 1
        (3, 5, 1),  # bottleneck 2
        (4, 6, 10),
        (5, 6, 10)
    ]
    g = Graph(from_list=edges)
    flow, g2 = g.maximum_flow(start=1, end=6)
    assert flow == 2, flow


def test_maximum_flow04():
    edges = [
        (1, 2, 10),
        (1, 3, 10),
        (2, 4, 1),  # bottleneck 1
        (2, 5, 1),  # bottleneck 2
        (3, 5, 1),  # bottleneck 3
        (3, 4, 1),  # bottleneck 4
        (4, 6, 10),
        (5, 6, 10)
    ]
    g = Graph(from_list=edges)
    flow, g2 = g.maximum_flow(start=1, end=6)
    assert flow == 4, flow


def test_maximum_flow05():
    edges = [
        (1, 2, 10),
        (1, 3, 1),
        (2, 3, 1)
    ]
    g = Graph(from_list=edges)
    flow, g2 = g.maximum_flow(start=1, end=3)
    assert flow == 2, flow


def test_maximum_flow06():
    edges = [
        (1, 2, 1),
        (1, 3, 1),
        (2, 4, 1),
        (3, 4, 1),
        (4, 5, 2),
        (5, 6, 1),
        (5, 7, 1),
        (6, 8, 1),
        (7, 8, 1)
    ]
    g = Graph(from_list=edges)
    flow, g2 = g.maximum_flow(start=1, end=8)
    assert flow == 2, flow
    assert set(g2.edges()) == set(edges)


def lecture_23_max_flow_problem():
    """ graph and stock from https://youtu.be/UtSrgTsKUfU """
    edges = [(1, 2, 8),
             (1, 3, 6),
             (2, 4, 5),
             (2, 5, 7),
             (3, 4, 6),
             (3, 5, 3),
             (4, 5, 4)]  # s,e,cost/unit
    g = Graph(from_list=edges)
    stock = {1: 6, 2: 0, 3: 4, 4: -5, 5: -5}  # supply > 0 > demand, stock == 0
    return g, stock


lec_23_optimum_cost = 81


def test_minimum_cost_flow_successive_shortest_path_unlimited():
    costs, inventory = lecture_23_max_flow_problem()
    mcf = minimum_cost_flow_using_successive_shortest_path
    total_cost, movements = mcf(costs, inventory)
    assert isinstance(movements, Graph)
    assert total_cost == lec_23_optimum_cost
    expected = [
        (1, 3, 6),
        (3, 4, 5),
        (3, 5, 5)
    ]
    for edge in movements.edges():
        expected.remove(edge)  # will raise error if edge is missing.
    assert expected == []  # will raise error if edge wasn't removed.


def test_minimum_cost_flow_successive_shortest_path_plenty():
    costs, inventory = lecture_23_max_flow_problem()
    capacity = Graph(from_list=[(s, e, lec_23_optimum_cost) for s, e, d in costs.edges()])
    mcf = minimum_cost_flow_using_successive_shortest_path
    total_cost, movements = mcf(costs, inventory, capacity)
    assert isinstance(movements, Graph)
    assert total_cost == lec_23_optimum_cost
    expected = [
        (1, 3, 6),
        (3, 4, 5),
        (3, 5, 5)
    ]
    for edge in movements.edges():
        expected.remove(edge)  # will raise error if edge is missing.
    assert expected == []  # will raise error if edge wasn't removed.


def test_minimum_cost_flow_successive_shortest_path_35_constrained():
    costs, inventory = lecture_23_max_flow_problem()
    capacity = Graph(from_list=[(s, e, lec_23_optimum_cost) for s, e, d in costs.edges()])
    capacity.add_edge(3, 5, 4)

    mcf = minimum_cost_flow_using_successive_shortest_path
    total_cost, movements = mcf(costs, inventory, capacity)
    assert isinstance(movements, Graph)
    assert total_cost == lec_23_optimum_cost - 3 - 6 + 8 + 7
    expected = [
        (1, 3, 5),
        (1, 2, 1),
        (2, 5, 1),
        (3, 5, 4),
        (3, 4, 5)
    ]
    for edge in movements.edges():
        expected.remove(edge)  # will raise error if edge is missing.
    assert expected == []  # will raise error if edge wasn't removed.


def test_minimum_cost_flow_successive_shortest_path_unlimited_excess_supply():
    costs, inventory = lecture_23_max_flow_problem()
    inventory[1] = 1000
    mcf = minimum_cost_flow_using_successive_shortest_path
    total_cost, movements = mcf(costs, inventory)
    assert isinstance(movements, Graph)
    assert total_cost == lec_23_optimum_cost
    expected = [
        (1, 3, 6),
        (3, 4, 5),
        (3, 5, 5)
    ]
    for edge in movements.edges():
        expected.remove(edge)  # will raise error if edge is missing.
    assert expected == []  # will raise error if edge wasn't removed.


