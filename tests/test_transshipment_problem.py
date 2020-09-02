from itertools import permutations
from time import process_time

from graph import Graph
from tests.test_graph import graph5x5

from graph.transshipment_problem import clondike_transshipment_problem, Train, schedule_rail_system
from graph.transshipment_problem import resolve2x3, resolve, bi_directional_bfs, bi_directional_progressive_bfs
from graph.transshipment_problem import check_user_input


def test_mining_train():
    """
    Assures that a train can schedule a number of in, out and in/out jobs
    using TSP.

    """
    g = clondike_transshipment_problem()
    assert isinstance(g, Graph)

    equipment_deliveries = [
        ("L-1", "L-1-1"),
        ("L-1", "L-1-2"),  # origin, destination
        ("L-1", "L-1-3"),
        ("L-1", "L-1-4")
    ]

    mineral_deliveries = [
        ("L-1-1", "L-1"),
        ("L-1-2", "L-1"),
        ("L-1-3", "L-1"),
        ("L-1-4", "L-1"),
    ]

    access_nodes = {"L-1", "L-1-1", "L-1-2", "L-1-3", "L-1-4"}

    train = Train(rail_network=g, start_location="L-1", access=access_nodes)

    s1 = train.schedule(equipment_deliveries)
    s2 = train.schedule(mineral_deliveries)
    s3 = train.schedule(equipment_deliveries[:] + mineral_deliveries[:])

    s1_expected = [
        ('L-1', 'L-1-1'), ('L-1', 'L-1-2'), ('L-1', 'L-1-3'), ('L-1', 'L-1-4')
    ]  # shortest jobs first.!

    s2_expected = [
        ('L-1-1', 'L-1'), ('L-1-2', 'L-1'), ('L-1-3', 'L-1'), ('L-1-4', 'L-1')
    ]  # shortest job first!

    s3_expected = [
        ('L-1', 'L-1-1'), ('L-1-1', 'L-1'),  # circuit 1
        ('L-1', 'L-1-2'), ('L-1-2', 'L-1'),  # circuit 2
        ('L-1', 'L-1-3'), ('L-1-3', 'L-1'),  # circuit 3
        ('L-1', 'L-1-4'), ('L-1-4', 'L-1')   # circuit 4
    ]  # shortest circuit first.

    assert s1 == s1_expected
    assert s2 == s2_expected
    assert s3 == s3_expected


def test_surface_mining_equipment_delivery():
    """
    Assures that equipment from the surface can arrive in the mine
    """
    g = clondike_transshipment_problem()

    equipment_deliveries = [
        ("Surface", "L-1-1"),
        ("Surface", "L-1-2"),  # origin, destination
    ]

    lift_access = {"Surface", "L-1", "L-2"}
    lift = Train(rail_network=g, start_location="Surface", access=lift_access)

    L1_access = {"L-1", "L-1-1", "L-1-2", "L-1-3", "L-1-4"}
    level_1_train = Train(rail_network=g, start_location="L-1", access=L1_access)

    assert lift_access.intersection(L1_access), "routing not possible!"

    schedule_rail_system(rail_network=g, trains=[lift, level_1_train],
                         jobs=equipment_deliveries)
    s1 = level_1_train.schedule()
    s2 = lift.schedule()

    s1_expected = [('L-1', 'L-1-1'), ('L-1', 'L-1-2')]
    s2_expected = [("Surface", "L-1"), ("Surface", "L-1")]

    assert s1 == s1_expected
    assert s2 == s2_expected


def test_double_direction_delivery():
    """

    Tests a double delivery schedule:

    Lift is delivering equipment into the mine as Job-1, Job-2
    Train is delivering gold out of the mine as Job-3, Job-4

    The schedules are thereby:

    Lift:   [Job-1][go back][Job-2]
    Train:  [Job-3][go back][Job-4]

    The combined schedule should thereby be:

    Lift:   [Job-1][Job-3][Job-2][Job-4]
    Train:  [Job-3][Job-1][Job-4][Job-2]

    which yields zero idle runs.

    """
    g = clondike_transshipment_problem()

    equipment_deliveries = [
        ("Surface", "L-1-1"),
        ("Surface", "L-1-2")
    ]

    mineral_deliveries = [
        ("L-1-1", "Surface"),
        ("L-1-2", "Surface")
    ]
    lift_access = {"Surface", "L-1", "L-2"}
    lift = Train(rail_network=g, start_location="Surface", access=lift_access)

    L1_access = {"L-1", "L-1-1", "L-1-2", "L-1-3", "L-1-4"}
    level_1_train = Train(rail_network=g, start_location="L-1", access=L1_access)

    assert lift_access.intersection(L1_access), "routing not possible!"

    schedule_rail_system(rail_network=g, trains=[lift, level_1_train],
                         jobs=equipment_deliveries + mineral_deliveries)
    s1 = level_1_train.schedule()
    s2 = lift.schedule()

    s1_expected = [('L-1', 'L-1-1'), ('L-1-1', 'L-1'), ('L-1', 'L-1-2'), ('L-1-2', 'L-1')]
    s2_expected = [('Surface', 'L-1'), ('L-1', 'Surface'), ('Surface', 'L-1'), ('L-1', 'Surface')]

    assert s1 == s1_expected
    assert s2 == s2_expected


def test_01():
    summary = {}

    items = [1, 2, 3,
             4, 5, 6]
    initial_state = "".join([str(i) for i in items])
    p, g = resolve2x3(initial_state, initial_state)

    for pz in permutations(items, 6):
        desire_state = "".join([str(i) for i in pz])
        d, p = g.states.shortest_path(initial_state, desire_state)

        d = len([i for i in p if isinstance(i, int)])

        if d not in summary:
            summary[d] = 1
        else:
            summary[d] += 1

    print("steps | freq")
    for k, v in sorted(summary.items()):
        print(k, v)
    print("it will never take more than", k, "steps to solve 2x3")
    tot, all = 0, sum(summary.values())
    for k, v in sorted(summary.items()):
        tot += v
        if tot > all / 2:
            break
    print("average is", k, "moves")
    # steps | freq
    # 0 1
    # 1 6
    # 2 28
    # 3 102
    # 4 231
    # 5 248
    # 6 100
    # 7 4
    # it will never take more than 7 steps to solve 2x3. Average is 4 moves


def test_simple_reroute():
    """ to loads on a collision path. """
    g = Graph()
    for s, e in [(1, 2), (2, 3)]:
        g.add_edge(s, e, 1, bidirectional=True)
    for s, e in [(1, 4), (4, 3)]:
        g.add_edge(s, e, 1, bidirectional=False)

    loads = {1: [1, 2, 3], 2: [3, 2, 1]}

    sequence = resolve(g, loads)
    assert sequence == [{1: (1, 4)},
                        {2: (3, 2)},
                        {2: (2, 1)},
                        {1: (4, 3)}]


def test_simple_reroute_2():
    """ to loads on a collision path. """
    g = Graph()
    for s, e in [(1, 2), (2, 3), (3, 4)]:
        g.add_edge(s, e, 1, bidirectional=True)
    for s, e in [(1, 5), (5, 6), (6, 4)]:
        g.add_edge(s, e, 1, bidirectional=False)

    loads = {1: [1, 2, 3, 4], 2: [4, 3, 2, 1]}

    sequence = resolve(g, loads)
    assert sequence == [{1: (1, 5)},
                        {2: (4, 3)},
                        {2: (3, 2)},
                        {2: (2, 1)},
                        {1: (5, 6)},
                        {1: (6, 4)}]


def test_simple_reroute_3():
    """ Loop with 6 nodes:
    1 <--> 2 <--> 3 <--> 4 <-- 5 <--> 6 <--> (1)
    """
    g = Graph()
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)
    g.del_edge(4, 5)

    loads = {1: [1, 2, 3], 2: [3, 2, 1]}

    sequence = resolve(g, loads)

    assert sequence == [{1: (1, 6)}, {2: (3, 2)}, {2: (2, 1)}, {1: (6, 5)}, {1: (5, 4)}, {1: (4, 3)}], sequence
    # assert sequence == [{1: (1, 6)}, {1: (6, 5)}, {2: (3, 2)}, {2: (2, 1)}, {1: (5, 4)}, {1: (4, 3)}], sequence


def test_simple_reroute_4():
    """
        1
       / \
      6---2
     / \ / \
    5 - 4 - 3
    """
    g = Graph()
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (2, 6), (2, 4), (6, 2)]
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)
    g.del_edge(4, 5)

    loads = {1: [1, 2, 3, 4],
             3: [3, 2, 1],
             6: [6, 2]}

    sequence = resolve(g, loads)
    assert sequence == [{1: (1, 2)}, {1: (2, 4)}, {3: (3, 2)}, {3: (2, 1)}, {6: (6, 2)}]

    g.del_edge(2, 4)

    sequence = resolve(g, loads)
    assert sequence == [{1: (1, 2)}, {3: (3, 4)}, {1: (2, 3)}, {3: (4, 2)}, {3: (2, 1)}, {6: (6, 2)}, {1: (3, 4)}]


def test_clockwise_rotation():
    """ A simple loop of 4 locations, where 3 loads need to move
    clockwise. """
    g = Graph()
    edges = [(1, 2), (2, 3), (3, 4), (4, 1), ]
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)

    loads = {1: [1, 2], 2: [2, 3], 3: [3, 4]}  # position 4 is empty.

    sequence = resolve(g, loads)

    assert sequence == [{3: (3, 4)},  # first move.
                        {2: (2, 3)},  # second move.
                        {1: (1, 2)}], sequence  # last move.


def test_small_gridlock():
    """ a grid lock is given, solver solves it."""
    g = Graph()
    edges = [
        (1, 2), (1, 4), (2, 3), (2, 5), (3, 6), (4, 5), (5, 6), (4, 7), (5, 8), (6, 9), (7, 8), (8, 9)
    ]
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)

    loads = {'a': [2, 1], 'b': [5, 2], 'c': [4, 3], 'd': [8], 'e': [1, 9]}
    check_user_input(g, loads)

    start = process_time()
    sequence = bi_directional_progressive_bfs(g, loads)
    end = process_time()
    print("duration:", end-start, "bi_directional_progressive_bfs")

    assert sequence == [{'a': (2, 3)},
                        {'a': (3, 6)},
                        {'a': (6, 9)},
                        {'b': (5, 2)},
                        {'c': (4, 5)},
                        {'c': (5, 6)},
                        {'c': (6, 3)},
                        {'a': (9, 6)},
                        {'a': (6, 5)},
                        {'a': (5, 4)},
                        {'a': (4, 7)},
                        {'e': (1, 4)},
                        {'e': (4, 5)},
                        {'e': (5, 6)},
                        {'e': (6, 9)},
                        {'a': (7, 4)},
                        {'a': (4, 1)}]
    start = process_time()
    sequence = bi_directional_bfs(g, loads)
    end = process_time()
    print("duration:", end - start, "bi_directional_bfs", flush=True)

    assert sequence == [{'b': (5, 6)},
                        {'b': (6, 3)},
                        {'c': (4, 5)},
                        {'c': (5, 6)},
                        {'e': (1, 4)},
                        {'a': (2, 1)},
                        {'e': (4, 5)},
                        {'b': (3, 2)},
                        {'c': (6, 3)},
                        {'e': (5, 6)},
                        {'e': (6, 9)}], sequence


def test_5x5_graph():
    g = graph5x5()
    loads = {'a': [6], 'b': [11, 1], 'c': [16, 2], 'd': [17, 4], 'e': [19, 5], 'f': [20, 3]}

    sequence = resolve(g, loads)

    assert sequence == [{'a': (6, 7)},
                        {'b': (11, 6)},
                        {'b': (6, 1)},
                        {'a': (7, 6)},
                        {'c': (16, 11)},
                        {'d': (17, 18)},
                        {'d': (18, 13)},
                        {'d': (13, 14)},
                        {'d': (14, 9)},
                        {'d': (9, 4)},
                        {'f': (20, 15)},
                        {'f': (15, 14)},
                        {'f': (14, 13)},
                        {'f': (13, 8)},
                        {'f': (8, 3)},
                        {'e': (19, 20)},
                        {'e': (20, 15)},
                        {'e': (15, 10)},
                        {'e': (10, 5)},
                        {'c': (11, 12)},
                        {'c': (12, 7)},
                        {'c': (7, 2)}], sequence


def test_2_trains():
    """
    two trains of loads are approaching each other.
    train 123 going from 1 to 14
    train 4567 going from 14 to 1.

    At intersection  4 train 123 can be broken apart and
    buffered, so that train 4567 can pass.

    The reverse (buffering train 4567) is not possible.
    """
    g = Graph()
    edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5), (4, 6), (4, 7), (4, 8),
        (5, 9), (6, 9), (7, 9), (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
    ]
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)

    loads = {
        41: [1, 2, 3, 4, 5, 9, 10, 11, 12],
        42: [2, 3, 4, 5, 9, 10, 11, 12, 13],
        43: [3, 4, 5, 9, 10, 11, 12, 13, 14],
        44: [11, 10, 9, 5, 4, 3, 2, 1],
        45: [12, 11, 10, 9, 5, 4, 3, 2],
        46: [13, 12, 11, 10, 9, 5, 4, 3],
        47: [14, 13, 12, 11, 10, 9, 5, 4],
    }

    sequence = resolve(g, loads)

    assert sequence == [{43: (3, 4)}, {43: (4, 6)}, {42: (2, 3)}, {42: (3, 4)}, {42: (4, 7)}, {41: (1, 2)},
                        {41: (2, 3)}, {41: (3, 4)}, {41: (4, 5)}, {44: (11, 10)}, {44: (10, 9)}, {44: (9, 8)},
                        {44: (8, 4)}, {44: (4, 3)}, {44: (3, 2)}, {44: (2, 1)}, {45: (12, 11)}, {45: (11, 10)},
                        {45: (10, 9)}, {45: (9, 8)}, {45: (8, 4)}, {45: (4, 3)}, {45: (3, 2)}, {46: (13, 12)},
                        {46: (12, 11)}, {46: (11, 10)}, {46: (10, 9)}, {46: (9, 8)}, {46: (8, 4)}, {46: (4, 3)},
                        {47: (14, 13)}, {47: (13, 12)}, {47: (12, 11)}, {47: (11, 10)}, {47: (10, 9)}, {47: (9, 8)},
                        {47: (8, 4)}, {43: (6, 9)}, {43: (9, 10)}, {43: (10, 11)}, {43: (11, 12)}, {43: (12, 13)},
                        {43: (13, 14)}, {42: (7, 9)}, {42: (9, 10)}, {42: (10, 11)}, {42: (11, 12)}, {42: (12, 13)},
                        {41: (5, 9)}, {41: (9, 10)}, {41: (10, 11)}, {41: (11, 12)}], sequence


def test_3_trains():
    """
    Two trains (abc & d) are going east. One train is going west (efgh).

    a-b-c--0-0-0--d--0--e-f-g-h
         \---0---/ \-0-/

    1-2-3--4-5-6--7--8---9-10-11-12
         \---13--/ \-14-/

    The solution is given by side stepping abc & d and letting efgh pass.
    """
    g = Graph()
    edges = [
        (3, 13), (13, 7), (7, 14), (14, 9)
    ]
    for a, b in zip(range(1, 12), range(2, 13)):
        edges.append((a, b))
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)

    loads = {
        'a': [1, 10], 'b': [2, 11], 'c': [3, 12], 'd': [8, 9],  # east bound
        'e': [9, 1], 'f': [10, 2], 'g': [11, 3], 'h': [12, 4]  # west bound
    }

    sequence = resolve(g, loads)
    assert sequence is not None
    

def test_4_trains():
    """ move through a bottleneck
    123 -0---0---456
         |   |
    789 -0---0---901

    456-123
    901-789

    901-123
    456-789

    789-123
    901-456
    """
    assert True


def test_random():
    """
    Generate random network
    Add loads at random
    Try to solve. Add to test suite if failed.

    Remember: The test suite can run forever. But the solver must be quick.
    """
    assert True
