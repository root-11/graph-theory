from sys import maxsize
from itertools import combinations
from collections import Counter
from statistics import stdev
from .base import BasicGraph
from bisect import insort


def tsp_branch_and_bound(graph):
    """
    Solve the traveling salesman's problem for the graph.
    :param graph: instance of class Graph
    :return: tour_length, path
    solution quality 100%
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")

    def lower_bound(graph, nodes):
        """Calculates the lower bound of distances for given nodes."""
        L = []
        edges = set()
        for n in nodes:
            L2 = [(d, e) for s, e, d in graph.edges(from_node=n) if e in nodes - {n}]
            if not L2:
                continue
            L2.sort()

            for d, n2 in L2:
                if (n2, n) in edges:  # Solution is not valid as it creates a loop.
                    continue
                else:
                    edges.add((n, n2))  # remember!
                    L.append((n, n2, d))
                    break

        return L

    global_lower_bound = sum(d for n, n2, d in lower_bound(graph, set(graph.nodes())))

    q = []
    all_nodes = set(graph.nodes())

    # create initial tree.
    start = graph.nodes()[0]
    for start, end, distance in graph.edges(from_node=start):
        lb = lower_bound(graph, all_nodes - {start})
        dist = sum(d for s, e, d in lb)
        insort(
            q,
            (
                distance + dist,  # lower bound of distance.
                -2,  # number of nodes in tour.
                (start, end),
            ),  # locations visited.
        )

    hit, switch, q2 = 0, True, []
    while q:  # walk the tree.
        d, _, tour = q.pop(0)
        tour_set = set(tour)

        if tour_set == all_nodes:
            if hit < len(all_nodes):  # to overcome premature exit.
                hit += 1
                insort(q2, (d, tour))
                continue
            else:
                d, tour = q2.pop(0)
                assert d >= global_lower_bound, "Solution not possible."
                return d, list(tour[:-1])

        remaining_nodes = all_nodes - tour_set

        for n2 in remaining_nodes:
            new_tour = tour + (n2,)

            lb_set = remaining_nodes - {n2}
            if len(lb_set) > 1:
                lb_dists = lower_bound(graph, lb_set)
                lb = sum(d for n, n2, d in lb_dists)
                new_lb = graph.distance_from_path(new_tour) + lb
            elif len(lb_set) == 1:
                last_node = lb_set.pop()
                new_tour = new_tour + (last_node, tour[0])
                new_lb = graph.distance_from_path(new_tour)
            else:
                raise Exception("bad logic!")

            insort(q, (new_lb, -len(new_tour), new_tour))

    return float("inf"), []  # <-- exit path if not solvable.


def tsp_greedy(graph):
    """
    Solves the traveling salesman's problem for the graph.
    Runtime approximation: seconds = 10**(-5) * (nodes)**2.31
    Solution quality: Range 98.1% - 100% optimal.
    :param graph: instance of class Graph
    :return: tour_length, path
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")

    t1 = _greedy(graph)
    d1 = graph.distance(t1)
    t2 = _opt2(graph, t1)
    d2 = graph.distance(t2)

    if d1 >= d2:  # shortest path wins.
        return d2, t2
    else:
        return d1, t1


# TSP 2023 ----


def _join_endpoints(endpoints, a, b):
    """Join segments [...,a] + [b,...] into one segment. Maintain `endpoints`.
    :param endpoints:
    :param a: node
    :param b: node
    :return:
    """
    a_seg, b_seg = endpoints[a], endpoints[b]
    if a_seg[-1] is not a:
        a_seg.reverse()
    if b_seg[0] is not b:
        b_seg.reverse()
    a_seg += b_seg
    del endpoints[a]
    del endpoints[b]
    endpoints[a_seg[0]] = endpoints[a_seg[-1]] = a_seg
    return a_seg


def _greedy(graph):
    c = combinations(graph.nodes(), 2)
    distances = [(graph.edge(a, b), a, b) for a, b in c if graph.edge(a, b)]
    distances.sort()

    new_segment = []
    endpoints = {n: [n] for n in graph.nodes()}
    for _, a, b in distances:
        if a in endpoints and b in endpoints and endpoints[a] != endpoints[b]:
            new_segment = _join_endpoints(endpoints, a, b)
            if len(new_segment) == len(graph.nodes()):
                break  # return new_segment

    if len(new_segment) != len(graph.nodes()):
        raise ValueError("there's an unconnected component in the graph.")
    return new_segment


def _opt1(graph, tour):
    """Iterative improvement based on relocation."""

    d_best = graph.distance(tour, return_to_start=True)
    p_best = tour
    L = tour[:]
    for i in range(len(tour)):
        tmp = L[:i] + L[i + 1 :]
        for j in range(len(tour)):
            L2 = tmp[:j] + [L[i]] + tmp[j:]
            d = graph.distance(L2, return_to_start=True)
            if d < d_best:
                d_best = d
                p_best = tuple(L2)

    tour = p_best

    # TODO: link swop
    # distances = [(graph.edge(tour[i], tour[i + 1]), i, i + 1) for i in range(len(tour)-1)]
    # distances.sort(reverse=True)  # longest link first.

    # for d1, i in distances:
    #     for j in range(len(tour)):
    #         if i == j:
    #             continue
    #         a, b = tour[i], tour[j]

    #     options = sorted([(d2, e) for _, e, d2 in graph.edges(from_node=a) if d2 < d1 and e != b])
    #     for d2, e in options:
    #         ix_e = tour.index(e)
    #     tmp = tour[32345678]

    # for i in range(len(tour)):
    #     tmp = p_best[:]
    #     for j in range(len(tour)):
    #         if i == j:
    #             continue

    return list(p_best)


def _opt2(graph, tour):
    """Iterative improvement based on 2 exchange."""

    def reverse_segment_if_improvement(graph, tour, i, j):
        """If reversing tour[i:j] would make the tour shorter, then do it."""
        # Given tour [...a,b...c,d...], consider reversing b...c to get [...a,c...b,d...]
        a, b, c, d = tour[i - 1], tour[i], tour[j - 1], tour[j % len(tour)]
        # are old links (ab + cd) longer than new ones (ac + bd)? if so, reverse segment.
        ab, cd, ac, bd = graph.edge(a, b), graph.edge(c, d), graph.edge(a, c), graph.edge(b, d)
        # if all are not None and improvement is shorter than previous ...
        if all((ab, cd, ac, bd)) and ab + cd > ac + bd:
            tour[i:j] = reversed(tour[i:j])  # ..retain the solution.
            return True

    def _zipwalk(tour):
        return [(tour[i - 1], tour[i]) for i in range(len(tour))]

    tour = list(tour)
    p0, d0 = tour[:], sum(graph.edge(a, b) for a, b in _zipwalk(tour))
    counter, inc = Counter(), 0
    while True:
        p0, d0 = tour[:], sum(graph.edge(a, b) for a, b in _zipwalk(tour))
        n = len(tour)
        # Return (i, j) pairs denoting tour[i:j] sub_segments of a tour of length N.
        g = ((i, i + length) for length in reversed(range(2, n)) for i in reversed(range(n - length + 1)))
        improvements = {reverse_segment_if_improvement(graph, tour, i, j) for (i, j) in g}

        d1 = sum(graph.edge(a, b) for a, b in _zipwalk(tour))
        if d1 < d0:
            d0 = d1
            p0 = tour[:]
        if improvements == {None} or len(improvements) == 0:
            return p0

        counter[tuple(tour)] += 1
        inc += 1
        if inc % 100 == 0:
            if stdev(counter.values()) > 2:  # the variance is exploding.
                return p0


def _opt3(graph, tour):
    """Iterative improvement based on 3 exchange."""

    def distance(a, b, graph=graph):
        return graph.edge(a, b, default=maxsize)

    def reverse_segment_if_better(tour, i, j, k):
        """If reversing tour[i:j] would make the tour shorter, then do it."""
        # Given tour [...A-B...C-D...E-F...]
        A, B, C, D, E, F = tour[i - 1], tour[i], tour[j - 1], tour[j], tour[k - 1], tour[k % len(tour)]
        d0 = distance(A, B) + distance(C, D) + distance(E, F)
        d1 = distance(A, C) + distance(B, D) + distance(E, F)
        d2 = distance(A, B) + distance(C, E) + distance(D, F)
        d3 = distance(A, D) + distance(E, B) + distance(C, F)
        d4 = distance(F, B) + distance(C, D) + distance(E, A)

        best = [(a, b) for a, b in zip([d0, d1, d2, d3, d4], ["d0", "d1", "d2", "d3", "d4"])]
        best.sort()
        _, index = best[0]
        if index == "d1":
            tour[i:j] = reversed(tour[i:j])
            return -d0 + d1
        elif index == "d2":
            tour[j:k] = reversed(tour[j:k])
            return -d0 + d2
        elif index == "d3":
            tmp = tour[j:k] + tour[i:j]
            tour[i:k] = tmp
            return -d0 + d3
        elif index == "d4":
            tour[i:k] = reversed(tour[i:k])
            return -d0 + d4
        else:
            return 0

    def all_segments(n: int):
        """Generate all segments combinations"""
        return ((i, j, k) for i in range(n) for j in range(i + 2, n) for k in range(j + 2, n + (i > 0)))

    while True:
        delta = 0
        for a, b, c in all_segments(len(tour)):
            delta += reverse_segment_if_better(tour, a, b, c)
        if delta >= 0:
            break
    return tour


def tsp_2023(graph):
    """
    TSP-2023 is the authors implementation of best practices discussed in Frontiers in Robotics and AI
    read more: https://www.frontiersin.org/articles/10.3389/frobt.2021.689908/full

    Args:
        graph (BasicGraph): fully connected subclass of BasicGraph
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")

    t = _greedy(graph)
    d = graph.distance(t)

    t1 = _opt1(graph, t)
    d1 = graph.distance(t1)

    t2 = _opt2(graph, t1)
    d2 = graph.distance(t2)

    t3 = _opt3(graph, t2)
    d3 = graph.distance(t3)

    L = [(d, t), (d1, t1), (d2, t2), (d3, t3)]
    L.sort()
    return L[0]
