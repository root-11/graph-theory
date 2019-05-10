from graph import Graph
from examples.optimization.transshipment_problem import clondike_transshipment_problem, Train, schedule_rail_system


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