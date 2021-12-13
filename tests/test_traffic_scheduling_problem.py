from time import process_time

from graph import Graph
from tests.test_graph import graph5x5

from graph.traffic_scheduling_problem import jam_solver
from graph.traffic_scheduling_problem import check_user_input, path_to_moves
from graph.traffic_scheduling_problem import hill_climb
from graph.traffic_scheduling_problem import moves_to_synchronous_moves
from graph.traffic_scheduling_problem import breadth_first_search


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
        4: (8, )
    }
    list_of_loads2 = list(check_user_input(g, loads_as_dict).values())

    assert list_of_loads1 == list_of_loads2


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

    loads = check_user_input(g, loads={1: [1, 3], 2: [3, 1]})
    d, p = hill_climb(g, loads)
    moves = path_to_moves(p)

    concurrent_moves = moves_to_synchronous_moves(moves, loads)
    assert concurrent_moves == [{1: (1, 4), 2: (3, 2)},
                                {2: (2, 1), 1: (4, 3)}]


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

    loads = check_user_input(g, loads={1: [1, 3], 2: [3, 1]})
    d, p = hill_climb(g, loads)
    moves = path_to_moves(p)
    concurrent_moves = moves_to_synchronous_moves(moves, loads)
    assert concurrent_moves == [{2: (3, 4), 1: (1, 2)},
                                {2: (4, 1), 1: (2, 3)}]
    assert moves == [{2: (3, 4)}, {1: (1, 2)}, {2: (4, 1)}, {1: (2, 3)}]


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

    loads = check_user_input(g, loads={1: [1, 3], 2: [3, 1]})
    d,p = hill_climb(g, loads)
    moves = path_to_moves(p)
    concurrent_moves = moves_to_synchronous_moves(moves, loads)
    assert concurrent_moves == [{2: (3, 4), 1: (1, 2)},
                                {2: (4, 1), 1: (2, 3)}]
    assert moves == [{2: (3, 4)}, {1: (1, 2)}, {2: (4, 1)}, {1: (2, 3)}]  # reverse of previous test.


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

    loads = check_user_input(g, loads={1: (1, [3], [2]),  # restriction 1 cannot travel over 2
                                       2: [3, 1]})
    d, p = hill_climb(g, loads)
    sequence = path_to_moves(p)

    assert sequence == [{2: (3, 2)}, {1: (1, 4)}, {1: (4, 3)}, {2: (2, 1)}]

    concurrent_moves = moves_to_synchronous_moves(sequence, loads)
    assert concurrent_moves == [{1: (1, 4), 2: (3, 2)},
                                {1: (4, 3), 2: (2, 1)}]


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

    loads = check_user_input(g, loads={1: (1, [3], [2]),  # restriction 1 cannot travel over 2
                                       2: (3, 1)})
    d, p = hill_climb(g, loads)
    sequence = path_to_moves(p)

    assert sequence == [{2: (3, 2)}, {1: (1, 4)}, {1: (4, 3)}, {2: (2, 1)}]
    concurrent_moves = moves_to_synchronous_moves(sequence, loads)
    assert concurrent_moves == [{1: (1, 4), 2: (3, 2)},
                                {2: (2, 1), 1: (4, 3)}]


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

    assert moves == [
        {2: (3, 4)},  # distance = 1, Load2 moves out of Load1's way.
        {1: (2, 3)},  # distance = 1
        {1: (3, 6)},  # distance = 3
        {2: (4, 3)},  # distance = 1, Load2 moves back onto it's starting point.
        {2: (3, 2)},  # distance = 1
        {2: (2, 1)}  # distance = 1
    ]  # total distance = (1+3)+(1+1+1+1) = 8


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

    assert moves == [
        {2: (3, 4)},  # distance 1
        {1: (2, 3)},  # distance 1
        {1: (3, 6)},  # distance 3
        {2: (4, 3)},  # distance 1
        {2: (3, 2)},  # distance 1
        {2: (2, 1)},  # distance 1
        {3: (5, 3)}  # distance 2
    ]  # total distance = (1+3)+(1+1+1+1)+(2) = 10


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

    assert moves == [
        {1: (2, 6)},  # 8
        {2: (3, 2)},  # 1
        {2: (2, 1)},  # 1
        {3: (4, 3)}   # 1
    ]  # total distance 11
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

    assert moves == [
        {2: (3, 5)},  # load 2 moves out of the way at cost 1
        {1: (2, 3)}, {1: (3, 6)},   # load 1 takes shortest permitted path.
        {2: (5, 3)}, {2: (3, 2)}, {2: (2, 1)},   # load 2 moves to destination.
        {3: (4, 3)}  # load 3 moves to destination.
    ]  # total distance 9


def test_energy_and_restrictions_2_load_high_detour_costs():
    """ See chart in example/images/tjs_problem_w_distance_restrictions.png """
    g = Graph()
    for sed in [
        (1, 2, 1),
        (1, 3, 70),  # this is the most direct route for load 2, but it is 70 long.
        (3, 5, 2),   # 5 is a dead end at higher cost than going (3,4)
        (3, 4, 1), (4, 6, 1),  # this could be shortest path for load 1, but
        # load 1 cannot travel over 4. If it could the path [2,3,4,6] would be 3 long.
        (2, 3, 1), (3, 6, 3),  # this is the shortest unrestricted route for load 1: [2,3,6] and it is 4 long.
        (2, 6, 50),  # this is the most direct route for load 1 [2,6] but it is 50 long.
    ]:
        g.add_edge(*sed, bidirectional=True)

    loads = {"A": (2, [6], [4]),
             "B": (3, 1)}

    moves = jam_solver(g, loads, synchronous_moves=False)

    assert moves == [
        {"B": (3, 4)},  # 2 moves into the dead end.
        {"A": (2, 3)},  # 1 moves into where 2 was.
        {"A": (3, 6)},  # 1 moves onto destination.
        {"B": (4, 3)},  # 2 moves back to origin along path [4,3,2,1]
        {"B": (3, 2)},
        {"B": (2, 1)}
    ]


def test_simple_reroute():
    """ to loads on a collision path. """
    g = Graph()
    for s, e in [(1, 2), (2, 3)]:
        g.add_edge(s, e, 1, bidirectional=True)
    for s, e in [(1, 4), (4, 3)]:
        g.add_edge(s, e, 1, bidirectional=False)

    loads = {1: [1, 3], 2: [3, 1]}

    concurrent_moves = jam_solver(g, loads, synchronous_moves=True)

    assert concurrent_moves == [{1: (1, 4), 2: (3, 2)},
                                {1: (4, 3), 2: (2, 1)}]


def test_simple_reroute_2():
    """ to loads on a collision path. """
    g = Graph()
    for s, e in [(1, 2), (2, 3), (3, 4)]:
        g.add_edge(s, e, 1, bidirectional=True)
    for s, e in [(1, 5), (5, 6), (6, 4)]:
        g.add_edge(s, e, 1, bidirectional=False)

    loads = {1: [1, 4], 2: [4, 1]}

    sequence = jam_solver(g, loads, synchronous_moves=False)

    assert sequence == [{1: (1, 5)}, {1: (5, 6)}, {2: (4, 3)}, {1: (6, 4)}, {2: (3, 2)}, {2: (2, 1)}]


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

    assert sequence == [{1: (1, 6)}, {1: (6, 5)}, {1: (5, 4)}, {2: (3, 2)}, {1: (4, 3)}, {2: (2, 1)}]

    # a = bfs_resolve(g, loads)
    # b = bi_directional_bfs(g, loads)
    # c = bi_directional_progressive_bfs(g, loads)
    # for d in a:
    #     b.remove(d)
    #     c.remove(d)
    # # all moves known in a have been removed from b and c.
    # assert b == c == []


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

    sequence = jam_solver(g, loads,synchronous_moves=False)
    assert sequence == [{1: (1, 2)}, {1: (2, 4)}, {3: (3, 2)}, {3: (2, 1)}, {6: (6, 2)}]

    g.del_edge(2, 4)

    sequence = jam_solver(g, loads,synchronous_moves=False)
    assert sequence == [{1: (1, 2)}, {3: (3, 4)}, {1: (2, 3)}, {3: (4, 2)}, {3: (2, 1)}, {6: (6, 2)}, {1: (3, 4)}]


def test_clockwise_rotation():
    """ A simple loop of 4 locations, where 3 loads need to move
    clockwise. """
    g = Graph()
    edges = [(1, 2), (2, 3), (3, 4), (4, 1), ]
    for s, e in edges:
        g.add_edge(s, e, 1, bidirectional=True)

    loads = {1: [1, 2], 2: [2, 3], 3: [3, 4]}  # position 4 is empty.

    sequence = jam_solver(g, loads, synchronous_moves=False)

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

    loads = check_user_input(g, {'a': [2, 1], 'b': [5, 2], 'c': [4, 3], 'd': [8], 'e': [1, 9]})

    start = process_time()
    distance, path = hill_climb(g, loads)
    end = process_time()

    assert distance == 19
    moves = path_to_moves(path)
    assert moves == [
        {'c': (4, 7)}, {'e': (1, 4)}, {'b': (5, 6)}, {'b': (6, 3)},
        {'e': (4, 1)}, {'c': (7, 4)}, {'c': (4, 5)}, {'e': (1, 4)},
        {'e': (4, 7)}, {'c': (5, 4)}, {'a': (2, 1)}, {'b': (3, 2)},
        {'c': (4, 5)}, {'e': (7, 4)}, {'c': (5, 6)}, {'e': (4, 5)},
        {'c': (6, 3)}, {'e': (5, 6)}, {'e': (6, 9)}]

    concurrent = moves_to_synchronous_moves(moves,loads)
    assert len(concurrent) == 8
    print("duration:", round(end - start,4), "| Hill climb | distance", distance, "| concurrent moves", len(concurrent))

    start = process_time()
    distance, path = breadth_first_search(g, loads)
    end = process_time()

    assert distance == 11
    moves = path_to_moves(path)
    assert moves == [
        {'b': (5, 6)}, {'b': (6, 3)}, {'c': (4, 5)}, {'c': (5, 6)},
        {'e': (1, 4)}, {'a': (2, 1)}, {'b': (3, 2)}, {'c': (6, 3)},
        {'e': (4, 5)}, {'e': (5, 6)}, {'e': (6, 9)}
    ]
    concurrent = moves_to_synchronous_moves(moves, loads)
    assert len(concurrent) == 6
    print("duration:", round(end - start,4), "| BFS | distance", distance, "| concurrent moves", len(concurrent))


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
    sequence = jam_solver(g, loads)

    assert sequence == [
        {'d': (5, 4), 'a': (8, 5)},  # d goes one step back. a moves forward to it's destination.
        {'a': (5, 9)},
        {'a': (9, 10)},
        {'a': (10, 11)},
        {'a': (11, 12), 'b': (7, 8)},  # b moves forward to it's destination.
        {'b': (8, 5)},
        {'b': (5, 9)},
        {'b': (9, 10)},
        {'b': (10, 11), 'c': (6, 5)},  # c moves forward.
        {'c': (5, 9)}, {'c': (9, 10), 'd': (4, 5)},  # d does a shortcut.
        {'d': (5, 9)}
    ]


def test_5x5_graph():
    g = graph5x5()
    loads = {'a': [6], 'b': [11, 1], 'c': [16, 2], 'd': [17, 4], 'e': [19, 5], 'f': [20, 3]}

    sequence = jam_solver(g, loads)
    assert sequence == [{'b': (11, 12), 'c': (16, 11), 'd': (17, 18), 'e': (19, 14), 'f': (20, 15)},
                        {'b': (12, 7), 'c': (11, 12), 'd': (18, 13), 'e': (14, 9), 'f': (15, 10)},
                        {'b': (7, 2), 'd': (13, 8), 'e': (9, 4), 'f': (10, 5)},
                        {'b': (2, 1), 'c': (12, 13), 'd': (8, 3)}, {'c': (13, 8)}, {'c': (8, 7)},
                        {'c': (7, 2), 'f': (5, 10), 'e': (4, 5), 'd': (3, 4)}, {'f': (10, 9)}, {'f': (9, 8)},
                        {'f': (8, 3)}]


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

    sequence = jam_solver(g, loads)

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

    sequence = jam_solver(g, loads)
    assert sequence is not None


def test_loop_9():
    g = Graph(
        from_list=[(a, b, 1) for a, b in zip(range(1, 8), range(2, 9))] + [(8, 1, 1)]
    )
    loads = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8]}
    solution = jam_solver(g, loads)
    expected = [{4: (7, 8)}, {3: (5, 6)}, {2: (3, 4)}, {1: (1, 2)}]
    for v in solution:
        assert v in expected, v


def test_loop_52():
    g = Graph(
        from_list=[
            (52, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1), (6, 7, 1), (7, 8, 1), (8, 9, 1),
            (9, 10, 1), (10, 11, 1), (11, 12, 1), (12, 13, 1), (13, 14, 1), (14, 15, 1), (15, 16, 1), (16, 17, 1),
            (17, 18, 1), (18, 19, 1), (19, 20, 1), (20, 21, 1), (21, 22, 1), (22, 23, 1), (23, 24, 1), (24, 25, 1),
            (25, 26, 1), (26, 27, 1), (27, 28, 1), (28, 29, 1), (29, 30, 1), (30, 31, 1), (31, 32, 1), (32, 33, 1),
            (33, 34, 1), (34, 35, 1), (35, 36, 1), (36, 37, 1), (37, 38, 1), (38, 39, 1), (39, 40, 1), (40, 41, 1),
            (41, 42, 1), (42, 43, 1), (43, 44, 1), (44, 45, 1), (45, 46, 1), (46, 47, 1), (47, 48, 1), (48, 49, 1),
            (49, 50, 1), (50, 51, 1), (51, 52, 1)
        ]
    )

    loads = {
        98: [52, 1], 55: [2, 3], 56: [3, 4], 57: [4, 5], 58: [5, 6], 59: [6, 7], 60: [7, 8], 61: [9, 10], 62: [10, 11],
        63: [11, 12], 64: [12, 13], 65: [14, 15], 66: [15, 16], 67: [16, 17], 68: [17, 18], 69: [18, 19], 70: [19, 20],
        71: [21, 22], 72: [22, 23], 73: [23, 24], 74: [24, 25], 75: [25, 26], 76: [26, 27], 77: [28, 29], 78: [29, 30],
        79: [30, 31], 80: [31, 32], 81: [32, 33], 82: [33, 34], 83: [35, 36], 84: [36, 37], 85: [37, 38], 86: [38, 39],
        87: [39, 40], 88: [40, 41], 89: [42, 43], 90: [43, 44], 91: [44, 45], 92: [45, 46], 93: [46, 47],
        94: [47, 48], 95: [49, 50], 96: [50, 51], 97: [51, 52]
    }

    solution = jam_solver(g, loads)
    expected = [{k: tuple(v)} for k, v in loads.items()]
    for v in solution:
        assert v in expected

    concurrent_moves = moves_to_synchronous_moves(solution, loads)
    assert concurrent_moves == [
        {98: (52, 1), 60: (7, 8), 59: (6, 7), 58: (5, 6), 57: (4, 5), 56: (3, 4), 55: (2, 3), 64: (12, 13),
         63: (11, 12), 62: (10, 11), 61: (9, 10), 70: (19, 20), 69: (18, 19), 68: (17, 18), 67: (16, 17), 66: (15, 16),
         65: (14, 15), 76: (26, 27), 75: (25, 26), 74: (24, 25), 73: (23, 24), 72: (22, 23), 71: (21, 22), 82: (33, 34),
         81: (32, 33), 80: (31, 32), 79: (30, 31), 78: (29, 30), 77: (28, 29), 88: (40, 41), 87: (39, 40), 86: (38, 39),
         85: (37, 38), 84: (36, 37), 83: (35, 36), 94: (47, 48), 93: (46, 47), 92: (45, 46), 91: (44, 45), 90: (43, 44),
         89: (42, 43), 97: (51, 52), 96: (50, 51), 95: (49, 50)}
    ], "something is wrong. All moves CAN happen at the same time."


def test_simple_failed_path():
    """ two colliding loads with no solution """
    g = Graph()
    for s, e in [(1, 2), (2, 3)]:
        g.add_edge(s, e, 1, bidirectional=True)

    loads = {1: [1, 3], 2: [3, 1]}

    sequence = jam_solver(g, loads)

    assert sequence is None


def test_incomplete_graph():
    """ two loads with an incomplete graph making the problem unsolvable """
    g = Graph()
    for s, e in [(1, 2), (2, 3)]:
        g.add_edge(s, e, 1, bidirectional=True)

    loads = {1: [1, 5], 2: [5, 1]}
    g.add_node(5)

    try:
        sequence = jam_solver(g, loads)
        assert False
    except ValueError as e:
        assert str(e) == f"No path found between {1} and {5}"

