from graph import Graph
from itertools import combinations

__description__ = """

MULTIPLE KNAPSACK PROBLEM

Definition:

Given a set of items, each with a weight and a value, determine the number of 
each item to include in a collection so that the total weight is less than or 
equal to a given limit and the total value is as large as possible. It derives 
its name from the problem faced by someone who is constrained by a fixed-size 
knapsack and must fill it with the most valuable items.[1]

Variations:
- If only one item fits into the knapsack, the problem is an assignment problem.

[1] https://en.wikipedia.org/wiki/Knapsack_problem

"""


def knapsack_solver(sacks_and_capacity, items_and_values):
    """
    The knapsack solver assumes that the graph is bi-partite and
    that one partition is one of more knapsacks, which has a value "size",
    and the other partition, the item.
    The edge(knapsack, item) has the value of the assignment.

    :param sacks: dict with sacks and capacities
    :param items: dict with items and values
    :return: graph with assignments.
    """
    if set(sacks_and_capacity).intersection(set(items_and_values)):
        raise ValueError("items and sacks have overlapping ids.")

    assignments = _combinatorial_knapsack_solver(sacks=sacks_and_capacity.copy(),
                                                 items=items_and_values)

    return assignments


def _combinatorial_knapsack_solver(sacks, items):
    """ ... Slow but thorough ...
    :param sacks: dict with sacks and capacities
    :param items: dict with items and values
    :return: graph with assignments
    """
    assert isinstance(items, dict)
    values = items.copy()  # I make a copy as I'm changing the structure.
    assert isinstance(sacks, dict)

    all_assignments = unique_powerset(list(values.values()))

    unassigned_values = [v for v in values.values()]

    assignment = Graph()
    for sack_id, capacity in sacks.items():

        candidate_solutions = [(capacity - sum(c), c)
                               for c in all_assignments
                               if sum(c) <= capacity]
        candidate_solutions.sort()
        for waste, combo in candidate_solutions:
            if not all(combo.count(i) <= unassigned_values.count(i) for i in combo):
                all_assignments.remove(combo)
                continue

            for value in combo:
                for item_id, v in values.items():
                    if v == value:
                        break
                del values[item_id]
                assignment.add_edge(sack_id, item_id, value=value)
                unassigned_values.remove(value)
            break
    return assignment


def unique_powerset(iterable):
    """
    The unique_powerset(iterable) returns the unique combinations of values
    when presented with repeated values:

        unique_powerset([1,1,1,2,2,3]) --> [
            (1,), (1, 1), (1, 1, 1), (2,), (2, 2), (3,),
            (1, 2), (1, 1, 2), (1, 1, 1, 2),
            (1, 2, 2), (1, 1, 2, 2), (1, 1, 1, 2, 2),
            (1, 3), (1, 1, 3), (1, 1, 1, 3),
            (2, 3), (2, 2, 3), (1, 2, 3),
            (1, 1, 2, 3), (1, 1, 1, 2, 3),
            (1, 2, 2, 3), (1, 1, 2, 2, 3),
            (1, 1, 1, 2, 2, 3)
        ] # 23 records

    This should be viewed in contrast to the powerset (see [1]) which would
    generate repeated values:

        powerset([1,1,1,2,2,3]) --> [
            (),
            (1,), (1,), (1,),  <-- duplicates
            (2,), (2,),
            (3,),
            (1, 1), (1, 1), (1, 1),
            (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2),
            (1, 3), (1, 3), (1, 3),
            ... cut for brevity ...
            (1, 1, 1, 2, 2), (1, 1, 1, 2, 3), (1, 1, 1, 2, 3),
            (1, 1, 2, 2, 3), (1, 1, 2, 2, 3), (1, 1, 2, 2, 3)
        ] # 63 records.

    The application of the unique_powerset is plenty, but most dominantly in
    the knapsack problem where a number of items must be matched to the
    capacity limit of the knapsack. Each combination of items determine the
    utilisation of the knapsack and the best should be selected, yet items
    of the same value can be substituted without further consideration

    As the number of duplicate values grow, the number of redundant options
    grows exponentially if using powerset.
    In the example above the powerset generates 63 vs the 23 unique options
    generated in the unique_powerset.

    The assertion set(powerset(iterable)) == unique_powerset(iterable) must
    always be true, and whilst the former method is available, powerset of
    any iterables longer than 20 items, become intolerable except for the most
    patient programmers.

    [1] https://docs.python.org/3/library/itertools.html#itertools-recipes

    :param iterable:
    :return: list of tuples
    """

    # first we summarize the iterable into blocks of identical values. Example:
    # [1,1,1,2,2,3] -->
    # d = {
    #     1: [[1],[1,1],[1,1,1]],
    #     2: [[2],[2,2]],
    #     3: [[3]]
    #     }
    d = {i: iterable.count(i) for i in set(iterable)}
    blocks = {i: [] for i in set(iterable)}
    for k, v in d.items():
        for i in range(1, v + 1):
            blocks[k].append([k] * i)

    # Next we generate the powersets of the unique values only:
    results = []
    for r in range(1, len(blocks) + 1):
        for clusters in combinations(blocks, r):
            # each 'cluster' is now an element from the powerset of the
            # unique elements --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

            # first we set indices for the accessing the first element in
            # the clusters values:
            c_index = [0 for _ in clusters]
            # this allows us to increment each index in values of each block.
            # Hereby c_index = [0,1,0] on the cluster (1,2,3) becomes [1,2,2,3].

            # next we set the upper limit to control the incremental iteration
            c_limit = [len(blocks[i]) for i in clusters]

            while not all(a == b for a, b in zip(c_index, c_limit)):
                # harvest combination
                result = []
                for idx, grp in enumerate(clusters):  # (1,2,3)
                    values = blocks[grp]  # v = 1:[[1],[1,1]. [1,1,1]]
                    value_idx = c_index[idx]  # [0,0,0]
                    value = values[value_idx]
                    result.extend(value)
                results.append(tuple(result))

                # update the indices:
                reset_idx = None
                for i in range(len(clusters)):  # counter position.
                    if c_index[i] < c_limit[i]:
                        c_index[i] += 1  # counter value

                    if c_index[i] == c_limit[i]:
                        reset_idx = i
                    else:
                        break

                # reset the preceding values in indices if the counter position
                # has incremented.
                if reset_idx is not None and reset_idx + 1 < len(clusters):
                    for j in range(reset_idx + 1):
                        c_index[j] = 0
    return results

