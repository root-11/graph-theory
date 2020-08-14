from itertools import permutations

from graph import Graph

from graph.transshipment_problem import clondike_transshipment_problem, Train, schedule_rail_system, resolve2x3, resolve


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
    edges1 = [(1, 2, 1), (2, 3, 1)]
    edges2 = [(1, 4, 1), (4, 3, 1)]
    for s, e, d in edges1:
        g.add_edge(s, e, d, bidirectional=True)
    for s, e, d in edges2:
        g.add_edge(s, e, d, bidirectional=False)

    loads = {1: [1, 2, 3], 2: [3, 2, 1]}

    sequence = resolve(g, loads)
    assert sequence == [{1: (1, 4)},
                        {2: (3, 2)},
                        {1: (4, 3)},
                        {2: (2, 1)}]


def test_api_1():
    """ A simple loop of 4 locations, where 3 loads need to move
    clockwise. """
    g = Graph()
    edges = [
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (4, 1, 1),
    ]
    for s, e, d in edges:
        g.add_edge(s, e, d, bidirectional=True)

    loads = {1: [1, 2], 2: [2, 3], 3: [3, 4]}  # position 4 is empty.

    sequence = resolve(g, loads)

    assert sequence == [{3: (3, 4)},  # first move.
                        {2: (2, 3)},  # second move.
                        {1: (1, 2)}]  # last move.


def test_api_1_1():
    """ a grid lock is given, solver solves it."""
    g = Graph()
    edges = [
        (1, 2, 1), (1, 4, 1), (2, 3, 1), (2, 5, 1), (3, 6, 1),
        (4, 5, 1), (5, 6, 1), (4, 7, 1), (5, 8, 1), (6, 9, 1),
        (7, 8, 1), (8, 9, 1)
    ]
    for s, e, d in edges:
        g.add_edge(s, e, d, bidirectional=True)

    loads = {1: [2, 1], 2: [5, 2], 3: [4, 3], 8: [8], 9: [1, 9]}

    sequence = resolve(g, loads)

    assert sequence == [{2: (5, 6)},
                        {2: (6, 3)},
                        {3: (4, 5)},
                        {3: (5, 6)},
                        {9: (1, 4)},
                        {1: (2, 1)},
                        {2: (3, 2)},
                        {3: (6, 3)},
                        {9: (4, 5)},
                        {9: (5, 6)},
                        {9: (6, 9)}]


def test_api_2():
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
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (4, 5, 1), (4, 6, 1), (4, 7, 1), (4, 8, 1),
        (5, 9, 1), (6, 9, 1), (7, 9, 1), (8, 9, 1),
        (9, 10, 1),
        (10, 11, 1),
        (11, 12, 1),
        (12, 13, 1),
        (13, 14, 1),
    ]
    for s, e, d in edges:
        g.add_edge(s, e, d, bidirectional=True)

    loads = {
        1: [1, 2, 3, 4, 5, 9, 10, 11, 12],
        2: [2, 3, 4, 5, 9, 10, 11, 12, 13],
        3: [3, 4, 5, 9, 10, 11, 12, 13, 14],
        4: [11, 10, 9, 5, 4, 3, 2, 1],
        5: [12, 11, 10, 9, 5, 4, 3, 2],
        6: [13, 12, 11, 10, 9, 5, 4, 3],
        7: [14, 13, 12, 11, 10, 9, 5, 4],
    }

    sequence = resolve(g, loads)

    assert sequence is None  # todo


def test_api_03():
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