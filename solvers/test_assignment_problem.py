from itertools import permutations

from graph import Graph
from solvers.assignment_problem import assignment_problem


def test_01_taxis_and_customers():
    """ Test where taxis are assigned to customers, so that the
    time to travel is minimised.

    Note that the travel time is negative, whereby the optimal
    assignment will have the minimal reponse time for all taxis
    to collect all customers.

    """
    taxis = [1, 2, 3]
    customers = [4, 5, 6]
    # relationships with distance as minutes apart.
    L = [
        (1, 4, -11),  # taxi 1, customer 4, 11 minutes apart.
        (1, 5, -23),
        (1, 6, -33),
        (2, 4, -14),
        (2, 5, -17),
        (2, 6, -34),
        (3, 4, -22),
        (3, 5, -19),
        (3, 6, -13)
    ]
    relationships = Graph(from_list=L)

    permutations_checked = 0
    for taxis_ in permutations(taxis, len(taxis)):
        for customers_ in permutations(customers, len(customers)):
            permutations_checked += 1
            print(permutations_checked, taxis_, customers_)
            assignment = assignment_problem(agents_and_tasks=relationships,
                                            agents=list(taxis_),
                                            tasks=list(customers_))
            assert set(assignment) == {(1, 4, -11), (2, 5, -17), (3, 6, -13)}
            assert sum(v for a, t, v in assignment) == sum([-11, -17, -13])
    print("The assignment problem solver is insensitive to initial conditions.")


def test_02_taxis_and_more_customers():
    """
    Like test_01, but with an additional customer who is attractive
    (value = -10)

    The same conditions exist as in test_01, but the least attractice
    customer (5) with value -17 is expected to be dropped.
    """
    taxis = [1, 2, 3]
    customers = [4, 5, 6, 7]
    # relationships with distance as minutes apart.
    L = [
        (1, 4, -11),  # taxi 1, customer 4, 11 minutes apart.
        (1, 5, -23),
        (1, 6, -33),
        (1, 7, -10),
        (2, 4, -14),
        (2, 5, -17),
        (2, 6, -34),
        (2, 7, -10),
        (3, 4, -22),
        (3, 5, -19),
        (3, 6, -13),
        (3, 7, -10)
    ]
    relationships = Graph(from_list=L)
    assignment = assignment_problem(agents_and_tasks=relationships,
                                    agents=list(taxis),
                                    tasks=list(customers))
    assert set(assignment) == {(1, 7, -10), (2, 4, -14), (3, 6, -13)}
    assert sum(v for a, t, v in assignment) > sum([-11, -17, -13])


def test_03_taxis_but_fewer_customers():
    """
    Like test_01, but with an additional taxi who is more attractive.

    We hereby expect that the new taxi (7) will steal the customer
    from 2.

    """
    taxis = [1, 2, 3, 7]
    customers = [4, 5, 6]
    # relationships with distance as minutes apart.
    L = [
        (1, 4, -11),  # taxi 1, customer 4, 11 minutes apart.
        (1, 5, -23),
        (1, 6, -33),
        (2, 4, -14),
        (2, 5, -17),
        (2, 6, -34),
        (3, 4, -22),
        (3, 5, -19),
        (3, 6, -13),
        (7, 4, -11),
        (7, 5, -11),
        (7, 6, -11),
    ]
    relationships = Graph(from_list=L)
    assignment = assignment_problem(agents_and_tasks=relationships,
                                    agents=list(taxis),
                                    tasks=list(customers))
    assert set(assignment) == {(1, 4, -11), (2, 5, -17), (7, 6, -11)}
    assert sum(v for a, t, v in assignment) > sum([-11, -17, -13])