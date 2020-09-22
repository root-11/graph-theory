import itertools
from bisect import insort

from graph import Graph


def lec_24_graph():
    """ Sample from https://www.youtube.com/watch?v=-cLsEHP0qt0

    Lecture series on Advanced Operations Research by
    Prof. G.Srinivasan, Department of Management Studies, IIT Madras.
    For more details on NPTEL visit http://nptel.iitm.ac.in

    """
    return Graph(
        from_list=[
            (1, 2, 10), (1, 3, 8), (1, 4, 9), (1, 5, 7),
            (2, 1, 10), (2, 3, 10), (2, 4, 5), (2, 5, 6),
            (3, 1, 8), (3, 2, 10), (3, 4, 8), (3, 5, 9),
            (4, 1, 9), (4, 2, 5), (4, 3, 8), (4, 5, 6),
            (5, 1, 7), (5, 2, 6), (5, 3, 9), (5, 4, 6)
        ]
    )


def test_tsp_brute_force():
    """ Generates all combinations of solutions """
    g = lec_24_graph()
    L = []
    shortest_tour = float('inf')
    for tour in itertools.permutations(g.nodes(), len(g.nodes())):
        d = g.distance_from_path(tour + (tour[0],))
        if d <= shortest_tour:
            insort(L, (d, tour))  # solutions are inserted by ascending distance.
            shortest_tour = d

    solutions = set()
    p1 = L[0][1]  # first solution == shortest tour.
    for d, t in L:
        if d == shortest_tour:
            t_reverse = tuple(list(t)[::-1])
            if any(
                    [g.same_path(t, p1),  # same path just offset in sequence.
                    g.same_path(t_reverse, p1)]  # same path reversed.
            ):
                solutions.add(t)
            else:
                raise AssertionError
    return solutions


lec_24_tsp_path = [1, 3, 4, 2, 5]
lec_24_valid_solutions = test_tsp_brute_force()
assert tuple(lec_24_tsp_path) in lec_24_valid_solutions


def test_greedy():
    g = lec_24_graph()
    d, tour = g.solve_tsp(method='greedy')
    assert tuple(tour) in lec_24_valid_solutions
    assert g.same_path(tour, lec_24_tsp_path)


def test_branch_and_bound():
    g = lec_24_graph()
    d, tour = g.solve_tsp(method='bnb')
    assert d == 34
    assert tuple(tour) in lec_24_valid_solutions


def test_bnb():
    g = Graph(from_list=[((755, 53), (282, 126), 478.60004178854814), ((755, 53), (559, 45), 196.16319736382766),
                         ((755, 53), (693, 380), 332.8257802514703), ((755, 53), (26, 380), 798.9806005154318),
                         ((755, 53), (229, 72), 526.3430440311718), ((755, 53), (655, 58), 100.12492197250393),
                         ((282, 126), (559, 45), 288.60006930006097), ((282, 126), (655, 58), 379.14772846477666),
                         ((282, 126), (229, 72), 75.66372975210778), ((282, 126), (755, 53), 478.60004178854814),
                         ((282, 126), (26, 380), 360.6272313622475), ((282, 126), (693, 380), 483.15318481823135),
                         ((655, 58), (559, 45), 96.87620966986684), ((655, 58), (26, 380), 706.6293229126569),
                         ((655, 58), (693, 380), 324.2344830520036), ((655, 58), (755, 53), 100.12492197250393),
                         ((655, 58), (282, 126), 379.14772846477666), ((655, 58), (229, 72), 426.2299848673249),
                         ((559, 45), (26, 380), 629.5347488423495), ((559, 45), (655, 58), 96.87620966986684),
                         ((559, 45), (693, 380), 360.8060420780118), ((559, 45), (755, 53), 196.16319736382766),
                         ((559, 45), (229, 72), 331.1027030998086), ((559, 45), (282, 126), 288.60006930006097),
                         ((26, 380), (229, 72), 368.8807395351511), ((26, 380), (693, 380), 667.0),
                         ((26, 380), (655, 58), 706.6293229126569), ((26, 380), (282, 126), 360.6272313622475),
                         ((26, 380), (755, 53), 798.9806005154318), ((26, 380), (559, 45), 629.5347488423495),
                         ((229, 72), (693, 380), 556.9201019895044), ((229, 72), (755, 53), 526.3430440311718),
                         ((229, 72), (655, 58), 426.2299848673249), ((229, 72), (559, 45), 331.1027030998086),
                         ((229, 72), (26, 380), 368.8807395351511), ((229, 72), (282, 126), 75.66372975210778),
                         ((693, 380), (755, 53), 332.8257802514703), ((693, 380), (229, 72), 556.9201019895044),
                         ((693, 380), (282, 126), 483.15318481823135), ((693, 380), (26, 380), 667.0),
                         ((693, 380), (559, 45), 360.8060420780118), ((693, 380), (655, 58), 324.2344830520036)]
              )
    d1, tour1 = g.solve_tsp('bnb')
    d2, tour2 = g.solve_tsp('greedy')
    assert d1 == d2

