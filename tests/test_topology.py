import time
from graph import Graph
from graph.critical_path import Task, critical_path, critical_path_minimize_for_slack
from graph.dag import phase_lines
from tests import profileit
from tests.test_graph import (
    graph02,
    graph3x3,
    graph_cycle_6,
    graph_cycle_5,
    fully_connected_4,
    mountain_river_map,
    sycamore,
    small_project_for_critical_path_method,
)


def test_subgraph():
    g = graph3x3()
    g2 = g.subgraph_from_nodes([1, 2, 3, 4])
    d = {
        1: {2: 1, 4: 1},
        2: {3: 1},
    }
    assert g2.is_subgraph(g)
    for k, v in d.items():
        for k2, d2 in v.items():
            assert g.edge(k, k2) == g2.edge(k, k2)

    g3 = graph3x3()
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
    g = graph3x3()
    assert not g.has_cycles()

def test_has_cycles():
    g = Graph(from_list=[(1, 2), (2, 3), (3, 1)])
    assert g.has_cycles()
    g.add_node(4)
    assert g.has_cycles()

    g.add_edge(4,5)
    assert g.has_cycles()
    g.add_edge(5,4)
    assert g.has_cycles()


def test_is_really_cyclic():
    g = Graph(from_list=[(1, 1, 1), (2, 2, 1)])  # two loops onto themselves.
    assert g.has_cycles()


def test_components():
    g = Graph(
        from_list=[
            (1, 2, 1),  # component 1
            (2, 1, 1),
            (3, 3, 1),  # component 2
            (4, 5, 1),
            (5, 6, 1),  # component 3
            (5, 7, 1),
            (6, 8, 1),
            (7, 8, 1),
            (8, 9, 1),
        ]
    )
    g.add_node(10)  # component 4
    components = g.components()
    assert len(components) == 4
    assert {1, 2} in components
    assert {3} in components
    assert {4, 5, 6, 7, 8, 9} in components
    assert {10} in components


def test_network_size():
    g = graph3x3()
    ns1 = g.network_size(n1=1)
    assert len(ns1) == 9  # all nodes.

    ns1_2 = g.network_size(n1=1, degrees_of_separation=2)
    assert all(i not in ns1_2 for i in [6, 8, 9])

    ns5 = g.network_size(n1=5)
    assert len(ns5) == 4  # all nodes downstream from 5 (plus 5 itself)

    ns9 = g.network_size(n1=9)
    assert len(ns9) == 1  # just node 9, as there are no downstream peers.


def test_network_size_when_fully_connected():
    """tests network size when the peer has already been seen during search."""
    g = fully_connected_4()
    ns = g.network_size(n1=1)
    assert len(ns) == len(g.nodes())


def test_phase_lines_with_loop():
    g = graph3x3()
    g.add_edge(9, 1)
    try:
        _ = g.phase_lines()
        assert False, "the graph is cyclic"
    except AttributeError:
        assert True


def test_phase_lines_with_inner_loop():
    g = graph3x3()
    g.add_edge(9, 2)
    try:
        _ = g.phase_lines()
        assert False, "the graph is cyclic"
    except AttributeError:
        assert True


def test_phase_lines_with_inner_loop2():
    g = graph3x3()
    g.add_edge(3, 2)
    try:
        _ = g.phase_lines()
        assert False, "the graph is cyclic"
    except AttributeError:
        assert True


def test_phase_lines_with_inner_loop3():
    g = graph3x3()
    g.add_edge(9, 10)
    g.add_edge(10, 5)
    try:
        _ = g.phase_lines()
        assert False, "the graph is cyclic"
    except AttributeError:
        assert True


def test_offset_phase_lines():
    """
    This test recreates a bug

    1
    |
    2
    |
    3   <---
    | /      |
    4   A    ^
    |   |    |
    5   B    |
    |___|   /
      6____/
      |
      7

    """
    g = Graph(
        from_list=[
            (1, 2, 1),
            (2, 3, 1),
            (3, 4, 1),
            (4, 5, 1),
            (5, 6, 1),
            (6, 7, 1),
            (6, 4, 1),
            ("a", "b", 1),
            ("b", 6, 1),
        ]
    )
    try:
        _ = g.phase_lines()
        assert False, "the graph is cyclic"
    except AttributeError:
        assert True


def test_phaselines():
    """
    1 +---> 3 +--> 5 +---> 6          [7]
                   ^       ^
      +------------+       |
      |
    2 +---> 4 +----------> +
    """
    g = Graph(
        from_list=[
            (1, 3, 1),
            (2, 4, 1),
            (2, 5, 1),
            (3, 5, 1),
            (4, 6, 1),
            (5, 6, 1),
        ]
    )
    g.add_node(7)

    p = g.phase_lines()
    assert set(g.nodes()) == set(p.keys())
    expects = {1: 0, 2: 0, 7: 0, 3: 1, 4: 1, 5: 2, 6: 3}
    assert p == expects, (p, expects)


def test_phaselines_for_ordering():
    """
    u1      u4      u2      u3
    |       |       |_______|
    csg     cs3       append
    |       |           |
    op1     |           op3
    |       |           |
    op2     |           cs2
    |       |___________|
    cs1         join
    |           |
    map1        map2
    |___________|
        save

    """
    L = [
        ("u1", "csg", 1),
        ("csg", "op1", 1),
        ("op1", "op2", 1),
        ("op2", "cs1", 1),
        ("cs1", "map1", 1),
        ("map1", "save", 1),
        ("u4", "cs3", 1),
        ("cs3", "join", 1),
        ("join", "map2", 1),
        ("map2", "save", 1),
        ("u2", "append", 1),
        ("u3", "append", 1),
        ("append", "op3", 1),
        ("op3", "cs2", 1),
        ("cs2", "join", 1),
    ]

    g = Graph(from_list=L)

    p = g.phase_lines()

    expected = {
        "u1": 0,
        "u4": 0,
        "u2": 0,
        "u3": 0,
        "csg": 1,
        "cs3": 1,
        "append": 1,
        "op1": 2,
        "op3": 2,
        "op2": 3,
        "cs2": 3,
        "cs1": 4,
        "join": 4,
        "map1": 5,
        "map2": 5,
        "save": 6,
    }

    assert p == expected, {(k, v) for k, v in p.items()} - {(k, v) for k, v in expected.items()}


def test_phaselines_for_larger_graph():
    g = mountain_river_map()
    start = time.time()
    p = g.phase_lines()
    g.has_cycles()
    end = time.time()

    # primary objective: correctness.
    assert len(p) == 253, len(p)

    # secondary objective: timeliness
    assert end - start < 1  # second.

    # third objective: efficiency.
    max_calls = 13000

    profiled_phaseline_func = profileit(phase_lines)

    calls, text = profiled_phaseline_func(g)
    if calls > max_calls:
        raise Exception(f"too many function calls: {text}")


def test_phaselines_for_sycamore():
    g = sycamore()
    start = time.time()
    p = g.phase_lines()
    end = time.time()

    # primary objective: correctness.
    assert len(p) == 1086, len(p)

    # secondary objective: timeliness
    assert end - start < 1  # second.

    # third objective: efficiency.
    max_calls = 17845

    profiled_phaseline_func = profileit(phase_lines)

    calls, text = profiled_phaseline_func(g)
    if calls > max_calls:
        raise Exception(f"too many function calls:\n{text}")
    
    t = list(g.topological_sort())
    assert t!=p


def test_sources():
    g = graph02()
    s = g.sources(5)
    e = {1, 2, 3}
    assert s == e

    s2 = g.sources(1)
    e2 = set()
    assert s2 == e2, s2

    s3 = g.sources(6)
    e3 = {1, 2, 3, 4, 5}
    assert s3 == e3

    s4 = g.sources(7)
    e4 = set()
    assert s4 == e4


def test_topological_sort():
    g = graph02()
    outcome = [n for n in g.topological_sort()]
    assert outcome == [1, 2, 7, 3, 4, 5, 6]

    outcome = [n for n in g.topological_sort(key=lambda x: -x)]
    assert outcome == [7, 2, 1, 4, 3, 5, 6]


def test_critical_path():
    g = small_project_for_critical_path_method()

    critical_path_length, schedule = critical_path(g)
    assert critical_path_length == 65
    expected_schedule = [
        Task("A", 10, 0, 0, 10, 10),
        Task("B", 20, 10, 10, 30, 30),
        Task("C", 5, 30, 30, 35, 35),
        Task("D", 10, 35, 35, 45, 45),
        Task("E", 20, 45, 45, 65, 65),
        Task("F", 15, 10, 25, 25, 40),
        Task("G", 5, 25, 40, 30, 45),
        Task("H", 15, 10, 30, 25, 45),
    ]

    for task in expected_schedule[:]:
        t2 = schedule[task.task_id]
        if task == t2:
            expected_schedule.remove(task)
        else:
            print(task, t2)
            raise Exception
    assert expected_schedule == []

    for tid, slack in {"F": 15, "G": 15, "H": 20}.items():
        task = schedule[tid]
        assert task.slack == slack

    # Note: By introducing a fake dependency from H to F.
    # the most efficient schedule is constructed, as all
    # paths become critical paths, e.g. where slack is
    # minimised as slack --> 0.

    g2 = critical_path_minimize_for_slack(g)
    critical_path_length, schedule = critical_path(g2)
    assert sum(t.slack for t in schedule.values()) == 0


def test_critical_path2():
    tasks = {"A": 1, "B": 10, "C": 1, "D": 5, "E": 2, "F": 1, "G": 1, "H": 1, "I": 1}
    dependencies = [
        ("A", "B"),
        ("B", "C"),
    ]
    for letter in "DEFGHI":
        dependencies.append(("A", letter))
        dependencies.append((letter, "C"))

    g = Graph()
    for n, d in tasks.items():
        g.add_node(n, obj=d)
    for n1, n2 in dependencies:
        g.add_edge(n1, n2)

    g2 = critical_path_minimize_for_slack(g)
    critical_path_length, schedule = critical_path(g2)
    assert sum(t.slack for t in schedule.values()) == 0, schedule


def test_critical_path3():
    g = small_project_for_critical_path_method()
    g.add_node("H", obj=5)  # reducing duration from 15 to 5, will produce more options.
    g2 = critical_path_minimize_for_slack(g)
    critical_path_length, schedule = critical_path(g2)
    assert critical_path_length == 65
    assert sum(t.slack for t in schedule.values()) == 30, schedule
