from collections import defaultdict
from functools import lru_cache
from heapq import heappop, heappush
from itertools import combinations, chain

from graph.graphs import BasicGraph
from graph.transform import all_pairs_shortest_paths
from graph.topology import has_path


def shortest_path(graph, start, end):
    """
    :param graph: class Graph
    :param start: start node
    :param end: end node
    :return: distance, path (as list),
             returns float('inf'), [] if no path exists.
    """
    assert isinstance(graph, BasicGraph)

    g = defaultdict(list)
    for n1, n2, dist in graph.edges():
        g[n1].append((dist, n2))

    q, visited, mins = [(0, start, ())], set(), {start: 0}
    while q:
        (cost, v1, path) = heappop(q)
        if v1 not in visited:
            visited.add(v1)
            path = (v1, path)

            if v1 == end:  # exit criteria.
                L = []
                while path:
                    v, path = path[0], path[1]
                    L.append(v)
                L.reverse()
                return cost, L

            for dist, v2 in g.get(v1, ()):
                if v2 in visited:
                    continue
                prev = mins.get(v2, None)
                next_node = cost + dist
                if prev is None or next_node < prev:
                    mins[v2] = next_node
                    heappush(q, (next_node, v2, path))
    return float("inf"), []


def breadth_first_search(graph, start, end):
    """ Determines the path from start to end with fewest nodes.
    :param graph: class Graph
    :param start: start node
    :param end: end node
    :return: path
    """
    assert isinstance(graph, BasicGraph)

    g = defaultdict(list)
    for n1, n2, dist in graph.edges():
        g[n1].append((dist, n2))

    q, visited, mins = [(0, start, ())], set(), {start: 0}
    while q:
        (cost, v1, path) = heappop(q)
        if v1 not in visited:
            visited.add(v1)
            path = (v1, path)

            if v1 == end:  # exit criteria.
                L = []
                while path:
                    v, path = path[0], path[1]
                    L.append(v)
                L.reverse()
                return cost, L  # <-- exit if end is found.

            for dist, v2 in g.get(v1, ()):
                if v2 in visited:
                    continue
                prev = mins.get(v2, None)
                next_node = cost + 1
                if prev is None or next_node < prev:
                    mins[v2] = next_node
                    heappush(q, (next_node, v2, path))
    return float("inf"), []  # <-- exit if end is not found.


def depth_first_search(graph, start, end):
    """
    Determines path from start to end using
    'depth first search' with backtracking.

    :param graph: class Graph
    :param start: start node
    :param end: end node
    :return: path as list of nodes.
    """
    assert start in graph, "start not in graph"
    assert end in graph, "end not in graph"
    q = [start]
    path = []
    visited = set()
    while q:
        n1 = q.pop()
        visited.add(n1)
        path.append(n1)
        if n1 == end:
            return path  # <-- exit if end is found.
        for n2 in graph.nodes(from_node=n1):
            if n2 in visited:
                continue
            q.append(n2)
            break
        else:
            path.remove(n1)
            while not q and path:
                for n2 in graph.nodes(from_node=path[-1]):
                    if n2 in visited:
                        continue
                    q.append(n2)
                    break
                else:
                    path = path[:-1]
    return None  # <-- exit if not path was found.


def tsp(graph):
    """
    Attempts to solve the traveling salesmans problem TSP for the graph.

    Runtime approximation: seconds = 10**(-5) * (points)**2.31
    Solution quality: Range 98.1% - 100% optimal.

    :param graph: instance of class Graph
    :return: tour_length, path
    """

    def shortest_links_first(graph):
        """ returns a list of (distance, node1, node2) with shortest on top."""
        c = combinations(graph.nodes(), 2)
        distances = [(graph.edge(a, b), a, b) for a, b in c if graph.edge(a, b)]
        distances.sort()
        return distances

    def join_endpoints(endpoints, a, b):
        """ Join segments [...,a] + [b,...] into one segment. Maintain `endpoints`.
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
        """ The TSP tour length WITH return to the starting point."""
        return sum(graph.edge(tour[i - 1], tour[i]) for i in range(len(tour)))

    def improve_tour(graph, tour):
        assert tour, "no tour to improve?"
        while True:
            improvements = {reverse_segment_if_improvement(graph, tour, i, j)
                            for (i, j) in sub_segments(len(tour))}
            if improvements == {None} or len(improvements) == 0:
                return tour

    @lru_cache()
    def sub_segments(n):
        """ Return (i, j) pairs denoting tour[i:j] sub_segments of a tour of length N."""
        return [(i, i + length) for length in reversed(range(2, n))
                for i in reversed(range(n - length + 1))]

    def reverse_segment_if_improvement(graph, tour, i, j):
        """If reversing tour[i:j] would make the tour shorter, then do it."""
        # Given tour [...A,B...C,D...], consider reversing B...C to get [...A,C...B,D...]
        a, b, c, d = tour[i - 1], tour[i], tour[j - 1], tour[j % len(tour)]
        # are old links (ab + cd) longer than new ones (ac + bd)? if so, reverse segment.
        if graph.edge(a, b) + graph.edge(c, d) > graph.edge(a, c) + graph.edge(b, d):
            tour[i:j] = reversed(tour[i:j])
            return True

    # The core TSP solver:
    assert isinstance(graph, BasicGraph), type(graph)

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


def shortest_tree_all_pairs(graph):
    """
       'minimize the longest distance between any pair'

    Note: This algorithm is not shortest path as it jumps
    to a new branch when it has exhausted a branch in the tree.
    :return: path
    """
    assert isinstance(graph, BasicGraph)
    g = all_pairs_shortest_paths(graph)
    assert isinstance(g, dict)

    distance = float('inf')
    best_starting_point = -1
    # create shortest path gantt diagram.
    for start_node in g.keys():
        if start_node in g:
            dist = sum(v for k, v in g[start_node].items())
            if dist < distance:
                best_starting_point = start_node
            # else: skip the node as it's isolated.
    g2 = g[best_starting_point]  # {1: 0, 2: 1, 3: 2, 4: 3}

    inv_g2 = {}
    for k, v in g2.items():
        if v not in inv_g2:
            inv_g2[v] = set()
        inv_g2[v].add(k)

    all_nodes = set(g.keys())
    del g
    path = []
    while all_nodes and inv_g2.keys():
        v_nearest = min(inv_g2.keys())
        for v in inv_g2[v_nearest]:
            all_nodes.remove(v)
            path.append(v)
        del inv_g2[v_nearest]
    return path


def all_paths(graph, start, end):
    """ Returns all paths from start to end.
    :param graph: instance of Graph
    :param start: node
    :param end: node
    :return: list of paths
    """
    assert isinstance(graph, BasicGraph)
    options = set(graph.nodes()) - {start, end}
    s = list(options)
    L = []
    # below generates the powerset of options:
    for combination in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)):
        path = [start] + list(combination) + [end]
        if has_path(graph, path):
            L.append(path)
    return L


def distance(graph, path):
    """ Calculates the distance for the path in graph
    :param graph: class Graph
    :param path: list of nodes
    :return: distance
    """
    assert isinstance(graph, BasicGraph)
    assert isinstance(path, list)
    cache = BasicGraph()
    path_length = 0
    for idx in range(len(path) - 1):
        n1, n2 = path[idx], path[idx + 1]
        d = cache.edge(n1, n2, default=None)
        if d is None:
            d, _ = shortest_path(graph, n1, n2)
            if d == float('inf'):
                return float('inf')
            cache.add_edge(n1, n2, value=d)
        path_length += d
    return path_length


def degree_of_separation(graph, n1, n2):
    """ Calculates the degree of separation between 2 nodes."""
    assert isinstance(graph, BasicGraph)
    assert n1 in graph.nodes()
    d, p = breadth_first_search(graph, n1, n2)
    return d

