from graph import Graph
from itertools import combinations
from fractions import Fraction

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

__all__ = ["knapsack_solver"]


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

    unassigned_values = [v for v in values.values()]

    assignment = Graph()

    max_capacity = max(sacks.values())
    item_combinations = []
    for i in unique_powerset(list(values.values()), max_value=max_capacity):
        if sum(i) <= max_capacity:
            item_combinations.append(i)

    for sack_id, capacity in sacks.items():
        candidate_solutions = [(capacity - sum(c), c)
                               for c in item_combinations
                               if sum(c) <= capacity]
        candidate_solutions.sort()
        for waste, combo in sorted(candidate_solutions):
            if not all(combo.count(i) <= unassigned_values.count(i) for i in combo):
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


def unique_powerset(iterable, max_value=None):
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

    :param iterable: list of values
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
            if max_value and k*i > max_value:
                break
            blocks[k].append([k] * i)

    if max_value is None:
        max_value = sum(iterable)+1

    # Next we generate the powersets of the unique values only:
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
                    values = blocks[grp]  # v = 1:[[1],[1,1],[1,1,1]]
                    value_idx = c_index[idx]  # [0,1,0]
                    value = values[value_idx]  # [1,1]
                    result.extend(value)
                if sum(result) < max_value:
                    yield tuple(result)
                else:  # if the sum of result exceeds the max value, we search
                    # through the index an increment the last pointer, and reset
                    # everything before that.
                    max_ix = 0
                    for ix in range(len(c_index)):
                        if c_index[ix] != 0:
                            max_ix = ix
                    if max_ix + 1 < len(c_index):
                        c_index[max_ix + 1] += 1
                        for j in range(max_ix + 1):
                            c_index[j] = 0
                        continue
                    else:
                        # however if we've reach the last pointer, we
                        # exit the while loop
                        break

                # if we reach this point, it'll be time to update the indices:
                reset_idx = None
                for i in range(len(clusters)):  # i = counter position.
                    if c_index[i] < c_limit[i]:
                        c_index[i] += 1  # counter value

                    if c_index[i] == c_limit[i]:
                        # then we've reached the limit of the counter, and
                        # should increment the pointer and reset everything
                        # before the pointer.
                        reset_idx = i
                    else:
                        break

                # reset the preceding values in indices if the counter position
                # has incremented.
                if reset_idx is not None and reset_idx + 1 < len(clusters):
                    for j in range(reset_idx + 1):
                        c_index[j] = 0


def auction_based_knapsack_solver(graph, options=None, start='start', end='end'):
    """
    An auction based solver for the knapsack problem, based only on transitive
    game theoretical perception of utility.

    :param graph: instance of class Graph with the following properties:

        1. "start" is the only node having in_degree == 0
        2. "end" is the only node having out_degree == 0
        3. The budget of each sack is declared as edge from "start" to "sack"
           with value = budget.
        4. The value of the assignment of "item" i to "sack" j is declared as
           the edge (j,i,value). Note that the value of i does not need to be
           the same for different sacks.
        5. The items all point to the "end" node.

    :param options: (optional) dict with mapping d[sack id = Graph()
        Example:
                +---> 3 +--+
        +-----> 1          |
        +       |          |
    start       +---> 4 +------> end
        +       |          |
        +-----> 2          |
                +---> 5 +--+

     Here the representation captures that the bidder expects to bid for one of
    the paths:

        [1,3], [1,4], [2,4], [2,5]

    :param start: node id of the 'start' node.
    :param end: node id of the 'end' node.

    NOTE: if options are used, then graph's start and end must be the same as
        the options start and end.

    :return: instance of Graph with assignments.

    How it works
    ---------------------------------------------------------------------------
    The solver assumes a multi-item auction where bidders (B) have different
    valuations of the items (I) and have different preferences, in so far that
    a combination of items may be preferable to another.

    An example could be a restaurent ordering items for recipes for a specials
    board: Each recipe needs to be complete, but only a limited number of
    recipes (maybe just one) needs to be complete.

    The data structure for representing these preferences are stored as a graph
    for each bidder:

                +---> 3 +--+
        +-----> 1          |
        +       |          |
    start       +---> 4 +------> end
        +       |          |
        +-----> 2          |
                +---> 5 +--+

    Here the representation captures that the bidder expects to bid for one of
    the paths:

        [1,3], [1,4], [2,4], [2,5]

    If the bidder finds that no path is possible during the auction, all bids
    are withdrawn.

    This provides generalisation for the cases where:

    (1) the valuation of the items is the same for each bidder is just a
    special case of the problem.

    (2) the bidder seeks to achieve a combination of items through the auction.

    By populating the graph with the auctioned value of each item, the total
    costs of each path become transparent and the bidder can make an informed
    choice for the moment, despite that information can be volatile as the
    auction progresses.

    The best choice, for the bidder is modeled as the shortest path from start
    to end (as this will represent the least cost in that moment).

    The auction is represented as a graph with the following properties:

        1. "start" is the only node having in_degree == 0
        2. "end" is the only node having out_degree == 0
        3. The budget of each sack is declared as edge from "start" to "sack"
           with value = budget.
        4. The value of the assignment of "item" i to "sack" j is declared as
           the edge (j,i,value). Note that the value of i does not need to be
           the same for different sacks.
        5. The items all point to the "end" node.

    # The auction ..
    ---------------------------------------------------------------------------
    The auction is started with an initial bid based on the valuation of the
    items as fractions.

    Consider for example a bidder (B1) who pursues the items 1,2,3 valued
    (relatively by the bidder) as [4,3,2]. The bidders budget (or capacity if
    using knapsack terms) is given as 7.

    Initial bids for the items are then given as:

        total valuation: 4+3+2 = 8.

        {1: 4/8, 2: 3/8, 3: 2/8} * 7 (budget) = {1: 28/8, 2: 21/8, 3: 14/8}

    This assures that if there only is one bidder, the items can be auctioned
    effectively in the first bid.

    Presuming another bidder (B2) pursue the items [2.3.4] with values [4,3,1]
    to the bidder and a budget of 5.
    Notice that the valuations are different between B1 and B2, and that there
    only is an overlap of interest for items 2 and 3.

    B2's initial bid is then given as:

        total valuation: 4+3+1 = 8.

        {2: 4/8. 3: 3/8, 4: 1/8} * 5 (budget) = {2: 20/8, 3: 15/8, 4: 5/8}

    The "invisible auctioneer" can now evaluate the initial bids as temporary
    assignments that set the bidding price.

    | item |   B1 |   B2 |      | B1 | B2 |      | item |
    +------+------+------+      +----+----+      +------+
    |   1  | 28/8 |    0 |  ==> |  1 |  0 !  ==> | 28/8 |
    |   2  | 21/8 | 20/8 |      |  1 |  0 |      | 21/8 |
    |   3  | 14/8 | 15/8 |      |  0 |  1 |      | 14/8 |
    |   4  |    0 |  5/8 |      |  0 |  1 |      |  5/8 |

    This application of relative valuations and budgets is perfectly aligned to
    J.V.Neumann and Oscar Morgensterns definition of utilty in the "Theory of
    Games and Economic Behaviour".

    In this assignment is a valid path for a bidder, the bidder remains passive
    until all bidders are passive - which would be the end of the auction.
    If the path is not valid (from start to end according to it's requirements)
    the bidder can increase bids using it's budget for auctioneable items, where
    it has not been assigned. Where it has been assigned, it must keep it's
    commitment.

    # Iterations:
    As iterations in the auction are set by the requirement to raise the
    auction value, E-complementary slackness scaling (Bertsekas) is guaranteed
    to prevent deadlock for the individual auction, but this does not guarantee
    prevention of deadlock for a the ability to obtain a critical set of items.

    There are therefore two options (A) and (B) available to each bidder:

    (A) As the bidder has information about prices and can calculate the lowest
    cost of all satisfying options (as the shortest path problem), the bidder
    can reassign budget from uncommitted bids and re-invest on items on the
    most cost-effective path.

    (B) If the bidder cannot find a path, it can drop out of all bids, whereby
    the assignment is reset to the next highest bidder (dutch auction principle)
    This allows B to free up funds from the initial bid to pursue an informed
    choice during the rest of the auction, and it prevents gridlock of sets of
    items.

    As a final notion each step in the auction is blind as all bids (and with-
    drawals) must be performed simultaneously: All bids are handled as closed
    bids into a pool and then processed before the next round is initiated. This
    guarantees that there is no advantage in being first or last in the bidding
    sequence.
    """
    if not isinstance(graph, Graph):
        raise TypeError("expected instance of class Graph, not {}".format(type(graph)))

    if options is None:
        options = {}
    else:
        if not isinstance(options, dict):
            raise TypeError("expected dict, not {}".format(type(options)))

        for sack_id, preference in options.items():
            if sack_id not in graph.nodes():
                raise ValueError("{} in options, but not in graph".format(sack_id))
            assert isinstance(preference, Graph)
            for node in preference.nodes():
                in_degree = len(preference.edges(to_node=node))
                out_degree = len(preference.edges(from_node=node))
                if in_degree == 0 or out_degree == 0:
                    if node in {start, end}:
                        continue
                    else:
                        raise ValueError("node {} in options doesn't use the start or end label".format(node))
                if node not in graph.nodes():
                    raise ValueError("{} found in preferences but doesn't exist in the graph.")

    assert start is not None
    assert end is not None
    if start not in graph.nodes():
        raise ValueError("""couldn't find node named "{}" """.format(start))
    if end not in graph.nodes():
        raise ValueError("""could find node named "{}" """.format(end))

    # ...input check completed.

    # start initial bid...
    auction = Graph()

    for s, bidder, budget in graph.edges(from_node=start):
        total_valuation = sum(v for b, i, v in graph.edges(from_node=bidder))
        for _, item, valuation in graph.edges(from_node=bidder):
            auction.add_edge(bidder, item, budget * Fraction(valuation, total_valuation))

    # start the auction.
    active_bidders = [b for b in graph.nodes(from_node=start)]
    while active_bidders:
        bidder = active_bidders.pop(0)

        # get assignment.
        assignments = []
        for item in graph.nodes(from_node=bidder):
            bids = []
            for b, i, v in graph.edges(to_node=item):
                bids.append((v, b))
            bids.sort(reverse=True)  # best bid is now on top.
            b, v = bids[0]
            assignments.append((item, b, v))  # item and value

        # what is left to invest?
        budget_left = graph.edges(to_node=bidder) - sum(v for i, b, v in assignments if b == bidder)

        # where can it be invested?
        for item, assigned_bidder, valuation in assignments:
            if assigned_bidder == bidder:
                continue  # bidder already owns the item.

        # what is the most sensible investment?
        # This question has the catch that others may attempt to invest
        # resources, in items that are already assigned and thereby present a
        # requirement for future investment. An option may therefor be NOT to
        # invest anything as the equivalent of waiting quietly at an auction,
        # in hope that others won't raise the bid.

        if bidder in options:  # does a preference graph exist?
            pass  # check if assignment is a valid solution.
        else:
            pass  # check what budget is left for bidding.

    return