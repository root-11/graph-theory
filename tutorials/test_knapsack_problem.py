import math
from tutorials.knapsack_problem import knapsack_solver, unique_powerset
import itertools


def test_1_d_stock_cutting_problem():
    """
    In operations research, the cutting-stock problem is the problem of cutting
    standard-sized pieces of stock material, such as paper rolls or sheet
    metal, into pieces of specified sizes while minimizing material wasted. It
    is an optimization problem in mathematics that arises from applications in
    industry.

    Illustration of one-dimensional cutting-stock problem

    A paper machine can produce an unlimited number of master (jumbo) rolls,
    each 5600 mm wide. The following 13 items must be cut, in the table below.

    The important thing about this kind of problem is that many different
    product units can be made from the same master roll, and the number of
    possible combinations is itself very large, in general, and not trivial to
    enumerate.

    The problem therefore is to find an optimum set of patterns of making
    product rolls from the master roll, such that the demand is satisfied and
    waste is minimized.

    [1] https://en.wikipedia.org/wiki/Cutting_stock_problem
    """
    master_roll_length = 5600  # mm

    bill_of_materials = [  # (width (mm), rolls)
        (1380, 22),
        (1520, 25),
        (1560, 12),
        (1710, 14),
        (1820, 18),
        (1880, 18),
        (1930, 20),
        (2000, 10),
        (2050, 12),
        (2100, 14),
        (2140, 16),
        (2150, 18),
        (2200, 20),
    ]
    lower_bound = sum(a * b for a, b in bill_of_materials)
    number_of_master_rolls = math.ceil(lower_bound / master_roll_length)
    items = [size for size, quantity in bill_of_materials for i in range(quantity)]

    limit = master_roll_length
    knapsacks = {"s{}".format(uid): limit for uid in range(number_of_master_rolls)}

    items = {"i{}".format(uid): length for uid, length in enumerate(items)}

    solution = knapsack_solver(sacks_and_capacity=knapsacks,
                               items_and_values=items)

    assert sum(v for a, b, v in solution.edges()) < lower_bound * 1.01

    # check that there's no breach of max length in assignments.
    for sack in knapsacks:
        assignments = solution.nodes(from_node=sack)
        if sum(items[item] for item in assignments) <= master_roll_length:
            pass
        else:
            raise ValueError
    # check that there's no duplication of assignments.
    for a, b in itertools.combinations(knapsacks, 2):
        assignments_1 = solution.nodes(from_node=a)
        assignments_2 = solution.nodes(from_node=b)
        assert set(assignments_1).isdisjoint(set(assignments_2))


def test_1_d_stock_cutting_problem_light():
    master_roll_length = 5600

    bill_of_materials = [  # (width (mm), rolls)
        (1710, 4),
        (2150, 2),
    ]
    lower_bound = sum(a * b for a, b in bill_of_materials)
    number_of_master_rolls = math.ceil(lower_bound / master_roll_length)
    items = [size for size, quantity in bill_of_materials for i in range(quantity)]

    limit = master_roll_length
    knapsacks = {"s{}".format(uid): limit for uid in range(number_of_master_rolls)}

    items = {"i{}".format(uid): length for uid, length in enumerate(items)}

    solution = knapsack_solver(sacks_and_capacity=knapsacks,
                               items_and_values=items)
    assert len(solution.edges()) == len(items)

    # check that there's no breach of max length in assignments.
    for sack in knapsacks:
        assignments = solution.nodes(from_node=sack)
        assert sum(items[item] for item in assignments) <= master_roll_length
    # check that there's no duplication of assignments.
    for a, b in itertools.combinations(knapsacks, 2):
        assignments_1 = solution.nodes(from_node=a)
        assignments_2 = solution.nodes(from_node=b)
        assert set(assignments_1).isdisjoint(set(assignments_2))


def test_unique_powerset():

    elements = [1, 1, 1, 2, 2, 3]
    pss = set()

    for r in range(1, len(elements)+1):
        for c in itertools.combinations(elements, r):
            pss.add(c)

    assert len(pss) == 23

    ups = unique_powerset(elements)

    assert len(ups) == 23
    assert set(pss) == set(ups)

    expected_results = [
        (1,),
        (1, 1),
        (1, 1, 1),
        (2,),
        (2, 2),
        (3,),
        (1, 2),
        (1, 1, 2),
        (1, 1, 1, 2),
        (1, 2, 2),
        (1, 1, 2, 2),
        (1, 1, 1, 2, 2),
        (1, 3),
        (1, 1, 3),
        (1, 1, 1, 3),
        (2, 3),
        (2, 2, 3),
        (1, 2, 3),
        (1, 1, 2, 3),
        (1, 1, 1, 2, 3),
        (1, 2, 2, 3),
        (1, 1, 2, 2, 3),
        (1, 1, 1, 2, 2, 3)
    ]

    assert ups == expected_results


def test_multiple_powersets():
    s = [
        [1, 2, 3],
        [1, 1, 2, 3],
        [1, 1, 2, 2, 3],
        [1, 1, 2, 2, 3, 3],
        [1, 2, 2, 3],
        [1, 2, 2, 3, 3],
        [1, 2, 3, 3],
    ]

    for i in s:
        ups = unique_powerset(i)
        assert ups is not None


def doall():
    L = [
        test_unique_powerset,
        test_multiple_powersets,
        test_1_d_stock_cutting_problem_light,
        test_1_d_stock_cutting_problem
    ]

    for f in L:
        print('starting', f.__name__)
        f()
        print(f.__name__, 'done')


if __name__ == "__main__":
    doall()
