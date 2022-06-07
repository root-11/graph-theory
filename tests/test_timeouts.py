from graph import Graph
from graph.traffic_scheduling_problem import jam_solver, NoSolution, UnSolvable,check_user_input, moves_to_synchronous_moves
from tests.utils import is_sequence_valid, is_matching
from tests.test_graph import graph5x5


def test_5x5_graph():
    g = graph5x5()
    loads = {'a': [6], 'b': [11, 1], 'c': [16, 2], 'd': [17, 4], 'e': [19, 5], 'f': [20, 3]}

    sequence = jam_solver(g, loads, return_on_first=True, timeout=30_000)
    assert is_sequence_valid(sequence, g)


def test_timeout():
    """ Timeout prevents all end states from being recorded, ensure that a solution is still found """
    edges = {1: {2: 1, 41: 2, 63: 2},
             41: {42: 1, 1: 2, 63: 2},
             65: {1: 2, 41: 2, 63: 2},
             2: {1: 1, 3: 1},
             3: {2: 1, 4: 1},
             4: {3: 1, 5: 1},
             5: {4: 1},
             42: {41: 1, 43: 1},
             43: {42: 1, 44: 1},
             44: {43: 1, 45: 1},
             45: {44: 1},
             63: {'pseudo_L48': 1, 'pseudo_L33': 1, 'pseudo_L35': 1, 'pseudo_L55': 1}}

    subgraph_2 = Graph(from_dict=edges)

    loads_for_jam_solver = {'L23': (41, [3, 4, 41, 44, 1, 2]),
                            'L48': (42, ['pseudo_L48']),
                            'L33': (43, ['pseudo_L33']),
                            'L8': (44, [3, 4, 41, 44, 1, 2]),
                            'L35': (45, ['pseudo_L35']),
                            'L5': (3, [3, 4, 41, 44, 1, 2]),
                            'L15': (4, [3, 4, 41, 44, 1, 2]),
                            'L55': (5, ['pseudo_L55'])}

    moves = jam_solver(graph=subgraph_2, loads=loads_for_jam_solver, timeout=5000, synchronous_moves=False)

    expected_moves = [{'L23': (41, 1)}, {'L48': (42, 41)}, {'L48': (41, 63)}, {'L48': (63, 'pseudo_L48')},
                      {'L33': (43, 42)}, {'L33': (42, 41)}, {'L33': (41, 63)}, {'L33': (63, 'pseudo_L33')},
                      {'L5': (3, 2)}, {'L15': (4, 3)}, {'L55': (5, 4)}, {'L23': (1, 41)}, {'L5': (2, 1)},
                      {'L15': (3, 2)}, {'L55': (4, 3)}, {'L23': (41, 42)}, {'L23': (42, 43)}, {'L5': (1, 41)},
                      {'L15': (2, 1)}, {'L55': (3, 2)}, {'L5': (41, 42)}, {'L15': (1, 41)}, {'L55': (2, 1)},
                      {'L55': (1, 63)}, {'L55': (63, 'pseudo_L55')}, {'L15': (41, 1)}, {'L5': (42, 41)},
                      {'L23': (43, 42)}, {'L15': (1, 2)}, {'L5': (41, 1)}, {'L23': (42, 41)}, {'L8': (44, 43)},
                      {'L35': (45, 44)}, {'L8': (43, 42)}, {'L35': (44, 43)}, {'L15': (2, 3)}, {'L5': (1, 2)},
                      {'L15': (3, 4)}, {'L5': (2, 3)}, {'L23': (41, 1)}, {'L23': (1, 2)}, {'L8': (42, 41)},
                      {'L35': (43, 42)}, {'L8': (41, 1)}, {'L35': (42, 41)}, {'L35': (41, 63)},
                      {'L35': (63, 'pseudo_L35')}]

    for index in range(5):
        assert moves[index] == expected_moves[index]


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

    solution = jam_solver(g, loads, return_on_first=True, timeout=1_000)
    assert is_sequence_valid(solution, g)

    loads2 = check_user_input(g, loads)
    concurrent_moves = moves_to_synchronous_moves(solution, loads2)
    assert concurrent_moves == [
        {98: (52, 1), 60: (7, 8), 59: (6, 7), 58: (5, 6), 57: (4, 5), 56: (3, 4), 55: (2, 3), 64: (12, 13),
         63: (11, 12), 62: (10, 11), 61: (9, 10), 70: (19, 20), 69: (18, 19), 68: (17, 18), 67: (16, 17), 66: (15, 16),
         65: (14, 15), 76: (26, 27), 75: (25, 26), 74: (24, 25), 73: (23, 24), 72: (22, 23), 71: (21, 22), 82: (33, 34),
         81: (32, 33), 80: (31, 32), 79: (30, 31), 78: (29, 30), 77: (28, 29), 88: (40, 41), 87: (39, 40), 86: (38, 39),
         85: (37, 38), 84: (36, 37), 83: (35, 36), 94: (47, 48), 93: (46, 47), 92: (45, 46), 91: (44, 45), 90: (43, 44),
         89: (42, 43), 97: (51, 52), 96: (50, 51), 95: (49, 50)}
    ], "something is wrong. All moves CAN happen at the same time."


def test_2_trains():
    """
    two trains of loads are approaching each other.
    train 123 going from 1 to 14
    train 4567 going from 14 to 1.

    At intersection  4 train 123 can be broken apart and
    buffered, so that train 4567 can pass.

    The reverse (buffering train 4567) is not possible.

    [1]--[2]--[3]--[4]--[5]--[9]--[10]--[11]--[12]--[13]--[14]
                    +---[6]---+
                    +---[7]---+
                    +---[8]---+
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
        41: [1, 12],
        42: [2, 13],
        43: [3, 14],
        44: [11, 1],
        45: [12, 2],
        46: [13, 3],
        47: [14, 4],
    }

    sequence = jam_solver(g, loads, return_on_first=True, timeout=180_000)
    assert is_sequence_valid(sequence, g)
    expected = [{43: (3, 4)}, {43: (4, 6)}, {42: (2, 3)}, {42: (3, 4)}, {42: (4, 7)}, {41: (1, 2)},
                {41: (2, 3)}, {41: (3, 4)}, {41: (4, 5)}, {44: (11, 10)}, {44: (10, 9)}, {44: (9, 8)},
                {44: (8, 4)}, {44: (4, 3)}, {44: (3, 2)}, {44: (2, 1)}, {45: (12, 11)}, {45: (11, 10)},
                {45: (10, 9)}, {45: (9, 8)}, {45: (8, 4)}, {45: (4, 3)}, {45: (3, 2)}, {46: (13, 12)},
                {46: (12, 11)}, {46: (11, 10)}, {46: (10, 9)}, {46: (9, 8)}, {46: (8, 4)}, {46: (4, 3)},
                {47: (14, 13)}, {47: (13, 12)}, {47: (12, 11)}, {47: (11, 10)}, {47: (10, 9)}, {47: (9, 8)},
                {47: (8, 4)}, {43: (6, 9)}, {43: (9, 10)}, {43: (10, 11)}, {43: (11, 12)}, {43: (12, 13)},
                {43: (13, 14)}, {42: (7, 9)}, {42: (9, 10)}, {42: (10, 11)}, {42: (11, 12)}, {42: (12, 13)},
                {41: (5, 9)}, {41: (9, 10)}, {41: (10, 11)}, {41: (11, 12)}]
    assert is_matching(expected, sequence), sequence


def test_3_trains():
    """
    Two trains (abc & d) are going east. One train is going west (efgh).

    a-b-c--0-0-0--d--0--e-f-g-h
         \---0---/ \-0-/

    1-2-3--4-5-6--7--8---9-10-11-12
         \---13--/ \-14-/

    The solution is given by side stepping abc (on 4,5,6) & d (on 8)
    and letting efgh pass on (12, 11, 10, 9, 14, 7, 13, 3, 2, 1)
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

    sequence = jam_solver(g, loads, return_on_first=True, timeout=40_000)
    assert sequence is not None


def test_simple_failed_path():
    """ two colliding loads with no solution """
    g = Graph()
    for s, e in [(1, 2), (2, 3)]:
        g.add_edge(s, e, 1, bidirectional=True)

    loads = {1: [1, 3], 2: [3, 1]}

    try:
        _ = jam_solver(g, loads, return_on_first=True, timeout=200)
        assert False, "The problem is unsolvable."
    except NoSolution:
        assert True


def test_incomplete_graph():
    """ two loads with an incomplete graph making the problem unsolvable """
    g = Graph()
    for s, e in [(1, 2), (2, 3)]:
        g.add_edge(s, e, 1, bidirectional=True)
    g.add_node(5)

    loads = {1: [1, 5], 2: [5, 1]}

    try:
        _ = jam_solver(g, loads, timeout=200)
        assert False, "There is no path."
    except UnSolvable as e:
        assert str(e) == 'load 1 has no path from 1 to 5'

