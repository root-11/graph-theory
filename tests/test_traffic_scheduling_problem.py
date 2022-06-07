from time import process_time
from graph import Graph


from graph.traffic_scheduling_problem import jam_solver
from graph.traffic_scheduling_problem import State
from graph.traffic_scheduling_problem import check_user_input
from graph.traffic_scheduling_problem import moves_to_synchronous_moves
from tests.utils import is_sequence_valid, is_matching


def test_data_loading():
    """ Checks the two acceptable data formats - happy path. """
    g = Graph(from_list=[
        (1, 2, 1.0), (2, 3, 0.2), (3, 4, 0.1), (4, 5, 0.5),
        (2, 7, 1.0), (2, 8, 0.5), (8, 9, 10)
    ])

    loads_as_list = [
        {'id': 1, 'start': 1, 'ends': 3},  # keyword prohibited is missing.
        {'id': 2, 'start': 2, 'ends': [3, 4, 5], 'prohibited': [7, 8, 9]},
        {'id': 3, 'start': 3, 'ends': [4, 5], 'prohibited': [2]},  # gateway to off limits.
        {'id': 4, 'start': 8}
    ]
    list_of_loads1 = list(check_user_input(g, loads_as_list).values())

    loads_as_dict = {
        1: (1, 3),  # start, end, None
        2: (2, [3, 4, 5], [7, 8, 9]),  # start, end(s), prohibited
        3: (3, [4, 5], [2]),
        4: (8,)
    }
    list_of_loads2 = list(check_user_input(g, loads_as_dict).values())

    assert list_of_loads1 == list_of_loads2


def test_check_concurrent_moves():
    A = [{2: (3, 4), 1: (1, 2)}, {2: (4, 1), 1: (2, 3)}]
    B = [{2: (3, 2), 1: (1, 4)}, {1: (4, 3), 2: (2, 1)}]
    assert is_matching(A, B)


def test_check_moves():
    A = [{2: (3, 4)}, {1: (1, 2)}, {2: (4, 1)}, {1: (2, 3)}]
    B = [{2: (3, 2)}, {1: (1, 4)}, {1: (4, 3)}, {2: (2, 1)}]
    assert is_matching(A, B)


def test_state_class():
    State(loads=(('A', 1), ('B', 2)))


def test_compact_bfs_problem():
    """
         [4]-->----+
          |        |
    [1]--[2]--[3]  v
          |        |
         [5]--<----+

    find the shortest path for the collision between load on [1] and load on [2]
    """
    g = Graph(from_list=[(1, 2, 1), (2, 3, 1), (4, 2, 1), (2, 5, 1), (4, 5, 3)])
    # edge 4,5 has distance 3, which is longer than path [4,2,5] which has distance 2.

    moves = jam_solver(g, loads={1: [1, 3], 2: [4, 5]})
    assert is_matching(moves, [{1: (1, 2)}, {1: (2, 3)}, {2: (4, 2)}, {2: (2, 5)}])


def test_hill_climb():
    """
    [1]<--->[2]<--->[3]
        \         /
         +->[4]->+  single direction!
    """
    g = Graph()
    for s, e in [(1, 2), (2, 3)]:
        g.add_edge(s, e, 1, bidirectional=True)
    for s, e in [(1, 4), (4, 3)]:
        g.add_edge(s, e, 1, bidirectional=False)

    loads = {1: [1, 3], 2: [3, 1]}
    moves = jam_solver(g,loads, synchronous_moves=True)
    expected = [{1: (1, 4), 2: (3, 2)}, {2: (2, 1), 1: (4, 3)}]
    assert is_matching(moves, expected), moves


def test_hill_climb_with_edge_weights():
    """  1       1
    [1]<--->[2]<--->[3]
        \ 3     1  /
         <->[4]<->
    All edges are weight 1, except 1<->4 which has weight 3.
    """
    g = Graph()
    for sed in [(1, 2, 1), (2, 3, 1),
                (1, 4, 3),  # <-- 3!
                (4, 3, 1)]:
        g.add_edge(*sed, bidirectional=True)

    loads={1: [1, 3], 2: [3, 1]}

    moves = jam_solver(g,loads)
    is_sequence_valid(moves, g)
    expected = [{2: (3, 4)}, {1: (1, 2)}, {2: (4, 1)}, {1: (2, 3)}]
    assert is_matching(moves, expected)

    concurrent_moves = jam_solver(g, loads, synchronous_moves=True)
    expected_conc_moves = [{2: (3, 4), 1: (1, 2)}, {2: (4, 1), 1: (2, 3)}]
    assert is_matching(concurrent_moves, expected_conc_moves)


def test_hill_climb_with_edge_different_weights():
    """ Same test as the previous, but this time edge 1<-->3 has weight 3,
    whilst the rest have weight 1.

         3       1
    [1]<--->[2]<--->[3]
        \ 1     1  /
         <->[4]<->
    """
    g = Graph()
    for sed in [(1, 2, 3),  # <-- 3!
                (2, 3, 1), (1, 4, 1), (4, 3, 1)]:
        g.add_edge(*sed, bidirectional=True)

    loads = {1: [1, 3], 2: [3, 1]}
    moves = jam_solver(g, loads)
    expected_moves = [{2: (3, 4)}, {1: (1, 2)}, {2: (4, 1)}, {1: (2, 3)}]  # reverse of previous test.
    assert is_matching(moves, expected_moves)

    concurrent_moves = jam_solver(g, loads, synchronous_moves=True)
    expected = [{2: (3, 4), 1: (1, 2)}, {2: (4, 1), 1: (2, 3)}]
    assert is_matching(concurrent_moves, expected)


def test_hill_climb_with_restrictions():
    """ Same problem as the previous, except that all weights are 1,
    and edge [1,4] and [4,3] are not bidirectional.
    and the only restriction is that load 1 cannot travel over node 2.

         1       1
    [1]<--->[2]<--->[3]
        \ 1     1  /
         -->[4]-->
    """
    g = Graph()
    for s, e in [(1, 2), (2, 3)]:
        g.add_edge(s, e, 1, bidirectional=True)
    for s, e in [(1, 4), (4, 3)]:
        g.add_edge(s, e, 1, bidirectional=False)

    loads = {1: (1, [3], [2]),  # restriction 1 cannot travel over 2
             2: [3, 1]}

    sequence = jam_solver(g, loads)

    expected_seq = [{2: (3, 2)}, {1: (1, 4)}, {1: (4, 3)}, {2: (2, 1)}]
    assert is_matching(sequence, expected_seq)

    concurrent_moves = jam_solver(g, loads, synchronous_moves=True)
    expected_conc_moves = [{1: (1, 4), 2: (3, 2)},{1: (4, 3), 2: (2, 1)}]
    assert is_matching(concurrent_moves, expected_conc_moves)


def test_hill_climb_with_restrictions_bidirectional():
    """ Same problem as the previous, except that all weights are 1,
    and the only restriction is that load 1 cannot travel over node 2.

         1       1
    [1]<--->[2]<--->[3]
        \ 1     1  /
         <->[4]<->
    """
    g = Graph()
    for s, e in [(1, 2), (2, 3), (1, 4), (4, 3)]:
        g.add_edge(s, e, 1, bidirectional=True)

    loads = {1: (1, [3], [2]),  # restriction 1 cannot travel over 2
             2: (3, 1)}

    sequence = jam_solver(g, loads)
    expected_seq = [{2: (3, 2)}, {1: (1, 4)}, {1: (4, 3)}, {2: (2, 1)}]
    assert is_matching(sequence, expected_seq)

    concurrent_moves = jam_solver(g, loads, synchronous_moves=True)
    assert concurrent_moves == [{1: (1, 4), 2: (3, 2)}, {2: (2, 1), 1: (4, 3)}]


def test_energy_and_restrictions_2_loads():
    """ See chart in example/images/tjs_problem_w_distance_restrictions.png

    NB: LENGTHS DIFFER FROM IMAGE!
    """
    g = Graph()
    for sed in [
        (1, 2, 1),
        (3, 5, 2),  # dead end.
        (1, 3, 7),  # this is the most direct route for load 2, but it is 7 long.
        (3, 4, 1), (4, 6, 1),  # this could be the shortest path for load 1, but
        # load 1 cannot travel over 4. If it could the path [2,3,4,6] would be 3 long.
        (2, 3, 1), (3, 6, 3),  # this is the shortest unrestricted route for load 1: [2,3,6] and it is 4 long.
        (2, 6, 8),  # this is the most direct route for load 1 [2,6] but it is 8 long.
    ]:
        g.add_edge(*sed, bidirectional=True)

    loads = {1: (2, [6], [4]),  # restriction load 1 cannot travel over 4
             2: (3, 1)}

    moves = jam_solver(g, loads, synchronous_moves=False)

    expected= [
        {2: (3, 4)},  # distance = 1, Load2 moves out of Load1's way.
        {1: (2, 3)},  # distance = 1
        {1: (3, 6)},  # distance = 3
        {2: (4, 3)},  # distance = 1, Load2 moves back onto it's starting point.
        {2: (3, 2)},  # distance = 1
        {2: (2, 1)}  # distance = 1
    ]  # total distance = (1+3)+(1+1+1+1) = 8
    assert is_matching(moves, expected)


def test_energy_and_restrictions_3_loads():
    """ See chart in example/images/tjs_problem_w_distance_restrictions.png
    NB: Lengths differ from image!
    """
    g = Graph()
    for sed in [

        (1, 2, 1),
        (3, 5, 2),  # dead end.
        (1, 3, 7),  # this is the most direct route for load 2, but it is 7 long.
        (3, 4, 1), (4, 6, 1),  # this could be the shortest path for load 1, but
        # load 1 cannot travel over 4. If it could the path [2,3,4,6] would be 3 long.
        (2, 3, 1), (3, 6, 3),  # this is the shortest unrestricted route for load 1: [2,3,6] and it is 4 long.
        (2, 6, 8),  # this is the most direct route for load 1 [2,6] but it is 8 long.
    ]:
        g.add_edge(*sed, bidirectional=True)

    loads = {1: (2, [6], [4]),
             2: (3, 1),
             3: (5, 3)}

    moves = jam_solver(g, loads, synchronous_moves=False)
    assert is_sequence_valid(moves, g)
    assert len(moves) == 7
    expected_moves = [
        {2: (3, 4)},  # distance 1
        {1: (2, 3)},  # distance 1
        {1: (3, 6)},  # distance 3
        {2: (4, 3)},  # distance 1
        {2: (3, 2)},  # distance 1
        {2: (2, 1)},  # distance 1
        {3: (5, 3)}  # distance 2
    ]  # total distance = (1+3)+(1+1+1+1)+(2) = 10
    assert is_matching(moves, expected_moves), moves


def test_energy_and_restrictions_3_loads_b():
    """ See chart in example/images/tjs_problem_w_distance_restrictions.png
    NB: Lengths differ from image!
    """
    g = Graph()
    for sed in [
        (1, 2, 1),
        (3, 5, 2),  # dead end.
        (1, 3, 7),  # this is the most direct route for load 2, but it is 7 long.
        (3, 4, 1), (4, 6, 1),  # this could be the shortest path for load 1, but
        # load 1 cannot travel over 4. If it could the path [2,3,4,6] would be 3 long.
        (2, 3, 1), (3, 6, 3),  # this is the shortest unrestricted route for load 1: [2,3,6] and it is 4 long.
        (2, 6, 8),  # this is the most direct route for load 1 [2,6] but it is 8 long.
    ]:
        g.add_edge(*sed, bidirectional=True)

    loads = {1: (2, [6], [4]),
             2: (3, 1),
             3: (4, 3)}  # Load 3 blocks load two from moving in here.

    moves = jam_solver(g, loads, synchronous_moves=False)
    atomic_moves = [
        {1: (2, 6)},  # 8
        {2: (3, 2)},  # 1
        {2: (2, 1)},  # 1
        {3: (4, 3)}  # 1
    ]  # total distance 11

    assert is_matching(moves, atomic_moves)
    # if load 2 would move to 5, the extra cost is 4, but load 1 could travel via [2,3,6] at cost 4.
    # However the distance LEFT for load 2 would be longer, whereby it is the lesser preferred solution.


def test_energy_and_restrictions_3_loads_c():
    """ See chart in example/images/tjs_problem_w_distance_restrictions.png
    NB: Lengths differ from image!
    """
    g = Graph()
    for sed in [
        (1, 2, 1),
        (3, 5, 1),  # dead end.
        (1, 3, 7),  # this is the most direct route for load 2, but it is 7 long.
        (3, 4, 1), (4, 6, 1),  # this could be the shortest path for load 1, but
        # load 1 cannot travel over 4. If it could the path [2,3,4,6] would be 3 long.
        (2, 3, 1), (3, 6, 3),  # this is the shortest unrestricted route for load 1: [2,3,6] and it is 4 long.
        (2, 6, 8),  # this is the most direct route for load 1 [2,6] but it is 8 long.
    ]:
        g.add_edge(*sed, bidirectional=True)

    loads = {1: (2, [6], [4]),
             2: (3, 1),
             3: (4, 3)}  # Load 3 blocks load two from moving in here.

    moves = jam_solver(g, loads, synchronous_moves=False)

    expected = [
        {2: (3, 5)},  # load 2 moves out of the way at cost 1
        {1: (2, 3)}, {1: (3, 6)},  # load 1 takes shortest permitted path.
        {2: (5, 3)}, {2: (3, 2)}, {2: (2, 1)},  # load 2 moves to destination.
        {3: (4, 3)}  # load 3 moves to destination.
    ]  # total distance 9
    assert is_matching(moves, expected), moves


def test_energy_and_restrictions_2_load_high_detour_costs():
    """ See chart in example/images/tjs_problem_w_distance_restrictions.png """
    g = Graph()
    for sed in [
        (1, 2, 1),
        (1, 3, 70),  # this is the most direct route for load 2, but it is 70 long.
        (3, 5, 2),  # 5 is a dead end at higher cost than going (3,4)
        (3, 4, 1), (4, 6, 1),  # this could be shortest path for load 1, but
        # load 1 cannot travel over 4. If it could the path [2,3,4,6] would be 3 long.
        (2, 3, 1), (3, 6, 3),  # this is the shortest unrestricted route for load 1: [2,3,6] and it is 4 long.
        (2, 6, 50),  # this is the most direct route for load 1 [2,6] but it is 50 long.
    ]:
        g.add_edge(*sed, bidirectional=True)

    loads = {"A": (2, [6], [4]),
             "B": (3, 1)}

    moves = jam_solver(g, loads, synchronous_moves=False, return_on_first=False)
    assert is_sequence_valid(moves, g)
    assert len(moves) == 6
    expected = [
        {"B": (3, 4)},  # 2 moves into the dead end.
        {"A": (2, 3)},  # 1 moves into where 2 was.
        {"A": (3, 6)},  # 1 moves onto destination.
        {"B": (4, 3)},  # 2 moves back to origin along path [4,3,2,1]
        {"B": (3, 2)},
        {"B": (2, 1)}
    ]
    assert is_matching(moves, expected), moves


def test_simple_reroute():
    """ to loads on a collision path. """
    g = Graph()
    for s, e in [(1, 2), (2, 3)]:
        g.add_edge(s, e, 1, bidirectional=True)
    for s, e in [(1, 4), (4, 3)]:
        g.add_edge(s, e, 1, bidirectional=False)

    loads = {1: [1, 3], 2: [3, 1]}

    concurrent_moves = jam_solver(g, loads, synchronous_moves=True)
    expected = [{1: (1, 4), 2: (3, 2)}, {1: (4, 3), 2: (2, 1)}]
    assert is_matching(concurrent_moves, expected), concurrent_moves


def test_simple_reroute_2():
    """ to loads on a collision path.

    [1]<-->[2]<-->[3]<-->[4]
     |                    ^
     +---->[5]-->[6]------+
    """
    g = Graph()
    for s, e in [(1, 2), (2, 3), (3, 4)]:
        g.add_edge(s, e, 1, bidirectional=True)
    for s, e in [(1, 5), (5, 6), (6, 4)]:
        g.add_edge(s, e, 1, bidirectional=False)

    loads = {1: [1, 4], 2: [4, 1]}

    sequence = jam_solver(g, loads, synchronous_moves=False)

    atomic_sequence = [{1: (1, 5)}, {1: (5, 6)}, {2: (4, 3)}, {1: (6, 4)}, {2: (3, 2)}, {2: (2, 1)}]
    assert is_matching(sequence, atomic_sequence)
    assert is_sequence_valid(sequence, g)


def test_simple_reroute_3():
    """ Loop with 6 nodes:
    1 <--> 2 <--> 3 <--> 4 <-- 5 <--> 6 <--> (1)
    """
    g = Graph()
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)]
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)
    g.del_edge(4, 5)

    loads = {1: [1, 3], 2: [3, 1]}

    sequence = jam_solver(g, loads, synchronous_moves=False)

    expected = [{1: (1, 6)}, {1: (6, 5)}, {1: (5, 4)}, {2: (3, 2)}, {1: (4, 3)}, {2: (2, 1)}]
    assert is_matching(sequence, expected)
    assert is_sequence_valid(sequence, g)


def test_shuffle():
    g = Graph()
    edges = [(1, 2), (2, 3), (3, 5), (3, 6), (5, 6), (6, 7), (7, 8)]
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)

    loads = {
        1: (1, 7),
        2: (2, [2, 5]),
        3: (8, [8, 5])
    }
    sequence = jam_solver(g, loads, synchronous_moves=False)

    expected = [
        {2: (2, 3)},
        {2: (3, 5)},
        {1: (1, 2)},
        {1: (2, 3)},
        {1: (3, 6)},
        {1: (6, 7)}]

    assert is_matching(sequence, expected), sequence


def test_shuffle2():
    g = Graph()
    edges = [(1, 2), (2, 3), (3, 5), (3, 6), (5, 6), (6, 7), (7, 8)]
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)
    loads = {
        1: (1, 7),
        2: (2, g.nodes()),
        3: (8, g.nodes())
    }

    sequence = jam_solver(g, loads, synchronous_moves=False, return_on_first=True)

    expected = [
        {2: (2, 3)},
        {2: (3, 5)},
        {1: (1, 2)},
        {1: (2, 3)},
        {1: (3, 6)},
        {1: (6, 7)}]

    assert is_matching(sequence, expected), sequence


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

    loads = {1: [1, 4],
             3: [3, 1],
             6: [6, 2]}

    sequence = jam_solver(g, loads, synchronous_moves=False)
    assert sequence == [{1: (1, 2)}, {1: (2, 4)}, {3: (3, 2)}, {3: (2, 1)}, {6: (6, 2)}]

    g.del_edge(2, 4)

    sequence = jam_solver(g, loads, synchronous_moves=False)

    expected = [{1: (1, 2)}, {3: (3, 4)}, {1: (2, 3)}, {3: (4, 2)}, {3: (2, 1)}, {6: (6, 2)}, {1: (3, 4)}]
    assert is_matching(sequence, expected)


def test_clockwise_rotation():
    """ A simple loop of 4 locations, where 3 loads need to move
    clockwise. """
    g = Graph()
    edges = [(1, 2), (2, 3), (3, 4), (4, 1), ]
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)

    loads = {1: [1, 2], 2: [2, 3], 3: [3, 4]}  # position 4 is empty.

    sequence = jam_solver(g, loads, synchronous_moves=False)

    expected = [{3: (3, 4)},  # first move.
                {2: (2, 3)},  # second move.
                {1: (1, 2)}]  # last move.
    assert is_matching(sequence, expected), sequence


def test_small_gridlock():
    """ a grid lock is given, solver solves it."""
    g = Graph()
    edges = [
        (1, 2), (1, 4), (2, 3), (2, 5), (3, 6), (4, 5), (5, 6), (4, 7), (5, 8), (6, 9), (7, 8), (8, 9)
    ]
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)

    loads = {'a': [2, 1], 'b': [5, 2], 'c': [4, 3], 'd': [8], 'e': [1, 9]}

    results = []

    # TRIAL - 1
    start = process_time()
    moves = jam_solver(g, loads)
    e = process_time() - start
    concurrent = jam_solver(g, loads, synchronous_moves=True)
    d = sum(len(d) for d in moves)
    results.append((e, d, len(concurrent)))

    assert d == 11, d
    expected = [{'b': (5, 6), 'c': (4, 5), 'e': (1, 4)},
                {'b': (6, 3), 'c': (5, 6), 'e': (4, 5), 'a': (2, 1)},
                {'b': (3, 2), 'c': (6, 3), 'e': (5, 6)},
                {'e': (6, 9)}]
    assert all(m in expected for m in moves), moves

    assert len(concurrent) == 4

    # TRIAL - 2
    start = process_time()
    moves = jam_solver(g, loads)
    e = process_time() - start
    concurrent = jam_solver(g, loads, synchronous_moves=True)
    d = sum(len(d) for d in moves)
    results.append((e, d, len(concurrent)))

    for e, d, c in results:
        print("duration:", round(e, 4), "| distance", d, "| concurrent moves", c)
    results.clear()


def test_snake_gridlock():
    """
    A bad route was given to train abcd, and now the train has gridlocked itself.

                    9 - 10 - 11 - 12
                    |
    1 - 2 - 3 - 4 - 5d-> 6c
                    ^    |
                    |    v
                    8a - 7b

    :return:
    """
    g = Graph()
    edges = [(a, b) for a, b in zip(range(1, 12), range(2, 13)) if (a, b) != (8, 9)]
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)
    g.add_edge(8, 5, 1, bidirectional=True)
    g.add_edge(5, 9, 1, bidirectional=True)

    loads = {'a': [8, 12], 'b': [7, 11], 'c': [6, 10], 'd': [5, 9]}
    sequence = jam_solver(g, loads, synchronous_moves=False, return_on_first=False)

    sync_moves = moves_to_synchronous_moves(sequence, check_user_input(g, loads))

    expected = [{'d': (5, 4)},  # d goes one step back.
                {'a': (8, 5)},  # a moves forward towards its destination.
                {'b': (7, 8)},  # b moves forward to it's destination.
                {'a': (5, 9)},
                {'b': (8, 5)},
                {'a': (9, 10)},
                {'b': (5, 9)},
                {'c': (6, 5)},  # c moves forward.
                {'a': (10, 11)},
                {'b': (9, 10)},
                {'c': (5, 9)},
                {'d': (4, 5)},
                {'a': (11, 12)},
                {'b': (10, 11)},
                {'c': (9, 10)},
                {'d': (5, 9)}]  # d does a left turn (shortcut).
    assert is_matching(sequence, expected)

    expected = [{'d': (5, 4), 'a': (8, 5), 'b': (7, 8)},
                {'a': (5, 9), 'b': (8, 5)},
                {'a': (9, 10), 'b': (5, 9), 'c': (6, 5)},
                {'a': (10, 11), 'b': (9, 10), 'c': (5, 9), 'd': (4, 5)},
                {'a': (11, 12), 'b': (10, 11), 'c': (9, 10), 'd': (5, 9)}]
    assert is_matching(expected, sync_moves), sync_moves


def test_loop_9():
    g = Graph(
        from_list=[(a, b, 1) for a, b in zip(range(1, 8), range(2, 9))] + [(8, 1, 1)]
    )
    loads = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8]}
    solution = jam_solver(g, loads, return_on_first=True)

    assert is_sequence_valid(solution, g)
    expected = [{4: (7, 8), 3: (5, 6), 2: (3, 4), 1: (1, 2)}]
    assert solution == expected

    sync_moves = jam_solver(g, loads, return_on_first=True, synchronous_moves=True)
    assert sync_moves == [{4: (7, 8), 3: (5, 6), 2: (3, 4), 1: (1, 2)}]

