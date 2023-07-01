from itertools import combinations
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

    def shortest_links_first(graph):
        """returns a list of (distance, node1, node2) with shortest on top."""
        c = combinations(graph.nodes(), 2)
        distances = [(graph.edge(a, b), a, b) for a, b in c if graph.edge(a, b)]
        distances.sort()
        return distances

    def join_endpoints(endpoints, a, b):
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

    def tsp_tour_length(graph, tour):
        """The TSP tour length WITH return to the starting point."""
        return sum(graph.edge(tour[i - 1], tour[i]) for i in range(len(tour)))
        # note to above: If there's an error it's probably because the graph isn't
        # fully connected.

    def improve_tour(graph, tour):
        assert tour, "no tour to improve?"
        n = len(tour)

        # Return (i, j) pairs denoting tour[i:j] sub_segments of a tour of length N.
        sub_segments = [(i, i + length) for length in reversed(range(2, n)) for i in reversed(range(n - length + 1))]
        cache = set()
        while True:
            tour_hash = hash(tuple(tour))
            if tour_hash in cache:
                return tour  # the search for optimization is repeating itself.
            else:
                cache.add(tour_hash)
                improvements = {reverse_segment_if_improvement(graph, tour, i, j) for (i, j) in sub_segments}
                if improvements == {None} or len(improvements) == 0:
                    return tour

    def reverse_segment_if_improvement(graph, tour, i, j):
        """If reversing tour[i:j] would make the tour shorter, then do it."""
        # Given tour [...A,B...C,D...], consider reversing B...C to get [...A,C...B,D...]
        a, b, c, d = tour[i - 1], tour[i], tour[j - 1], tour[j % len(tour)]
        # are old links (ab + cd) longer than new ones (ac + bd)? if so, reverse segment.
        A, B, C, D = graph.edge(a, b), graph.edge(c, d), graph.edge(a, c), graph.edge(b, d)
        # if all are not None and improvement is shorter than previous ...
        if all((A, B, C, D)) and A + B > C + D:
            tour[i:j] = reversed(tour[i:j])  # ..retain the solution.
            return True

    # The core TSP solver
    # -----------------------
    # 1. create a path using greedy algorithm (picks nearest peer)
    new_segment = []
    endpoints = {n: [n] for n in graph.nodes()}
    L = shortest_links_first(graph)
    for _, a, b in L:
        if a in endpoints and b in endpoints and endpoints[a] != endpoints[b]:
            new_segment = join_endpoints(endpoints, a, b)
            if len(new_segment) == len(graph.nodes()):
                break  # return new_segment
    assert len(new_segment) == len(graph.nodes()), "there's an unconnected component."
    first_tour = new_segment[:]
    first_path_length = tsp_tour_length(graph, first_tour)

    # 2. run improvement on the created path.
    improved_tour = improve_tour(graph, new_segment)
    assert set(graph.nodes()) == set(improved_tour)

    second_path_length = tsp_tour_length(graph, improved_tour)

    assert first_path_length >= second_path_length, "first path was better than improved tour?! {} {}".format(
        first_path_length, second_path_length
    )

    return second_path_length, improved_tour


def tsp_2023(graph):
    """
    TSP-2023 is the authors implementation of best practices discussed in Frontiers in Robotics and AI
    read more: https://www.frontiersin.org/articles/10.3389/frobt.2021.689908/full

    Args:
        graph (BasicGraph): fully connected subclass of BasicGraph
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    raise NotImplementedError()
