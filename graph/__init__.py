from functools import lru_cache
from itertools import combinations
from collections import defaultdict
from heapq import heappop, heappush


__all__ = ['Graph']


class Graph(object):
    """
    Graph is the base graph that all methods use.

    """
    def __init__(self, nodes=None, links=None):
        if nodes is None:
            nodes = {}
        self.nodes = nodes
        if links is None:
            links = {}
        self.links = links
        self._max_length = 0
        self._edges_cache = None

    def __getitem__(self, item):
        return self.links.__getitem__(item)

    def edges(self):
        """
        :return: list of edges (n1, n2, distance)
        """
        if self._edges_cache:
            return self._edges_cache[:]

        L = []
        for n1 in self.links:
            for n2 in self.links[n1]:
                L.append((n1, n2, self.links[n1][n2]))
        return L

    def add_node(self, node_id):
        self.nodes[node_id] = 1

    def add_link(self, node1, node2, distance=1, bidirectional=False):
        """
        :param node1: hashable node
        :param node2: hashable node
        :param distance: numeric value.
        :param bidirectional: boolean.
        """
        self._edges_cache = None  #

        assert isinstance(distance, (float, int))
        self.add_node(node1)
        self.add_node(node2)
        if node1 not in self.links:
            self.links[node1] = {}
        if node2 not in self.links:
            self.links[node2] = {}
        self.links[node1][node2] = distance
        if distance > self._max_length:
            self._max_length = distance
        if bidirectional:
            self.links[node2][node1] = distance

    def update_from_dict(self, dictionary):
        """
        Creates graph from dictionary
        :param dictionary:

        d = {1: {2: 10, 3: 5},
             2: {4: 1, 3: 2},
             3: {2: 3, 4: 9, 5: 2},
             4: {5: 4},
             5: {1: 7, 4: 6}}
        G = from_dict(d)

        :return: class Graph.
        """
        for n1 in dictionary:
            for n2 in dictionary[n1]:
                self.add_link(n1, n2, dictionary[n1][n2])

    # methods:
    def shortest_path(self, start, end):

        return shortest_path(graph=self, start=start, end=end)

    def distance_from_path(self, path):
        return distance(graph=self, path=path)

    def solve_tsp(self):
        return tsp(self)

    def subgraph_from_nodes(self, nodes):
        return subgraph(graph=self, nodes=nodes)

    def same_path(self, p1, p2):
        return same(p1, p2)


def shortest_path(graph, start, end):
    """
    :param graph: class Graph
    :param start: start node
    :param end: end node
    :return: distance, path (as list),
             returns float('inf'), [] if no path exists.
    """
    assert isinstance(graph, Graph)

    g = defaultdict(list)
    for n1, n2, dist in graph.edges():
        g[n1].append((dist, n2))

    q, visited, mins = [(0, start, ())], set(), {start: 0}
    while q:
        (cost, v1, path) = heappop(q)
        if v1 not in visited:
            visited.add(v1)
            path = (v1, path)
            if v1 == end:
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
                next = cost + dist
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf"), []


def distance(graph, path):
    """ Calculates the distance for the path in graph
    :param graph: class Graph
    :param path: list of nodes
    :return: distance
    """
    assert isinstance(graph, Graph)
    assert isinstance(path, list)
    d = 0
    for idx in range(len(path)-1):
        n1, n2 = path[idx], path[idx+1]
        d += graph.links[n1][n2]
    return d


def subgraph(graph, nodes):
    """ Creates a subgraph as a copy from the graph
    :param graph: class Graph
    :param nodes: list of nodes
    :return: new instance of Graph.
    """
    assert isinstance(graph, Graph)
    assert isinstance(nodes, list)
    assert all(n1 in graph.nodes for n1 in nodes)
    G = Graph()
    for n1 in nodes:
        G.add_node(n1)
        for n2 in graph[n1]:
            G.add_link(n1, n2, graph[n1][n2])
    return G


def same(path1, path2):
    """ Compares two paths to verify whether they're the same.
    :param path1: list of nodes.
    :param path2: list of nodes.
    :return: boolean.
    """
    start1 = path2.index(path1[0])
    checks = [
        path1[:len(path1) - start1] == path2[start1:],
        path1[len(path1) - start1:] == path2[:start1]
    ]
    if all(checks):
        return True
    return False


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
        c = combinations(graph.nodes, 2)
        distances = [(graph.links[a][b], a, b) for a, b in c]
        distances.sort()
        return distances

    def join_endpoints(endpoints, A, B):
        "Join segments [...,A] + [B,...] into one segment. Maintain `endpoints`."
        Aseg, Bseg = endpoints[A], endpoints[B]
        if Aseg[-1] is not A: Aseg.reverse()
        if Bseg[0] is not B: Bseg.reverse()
        Aseg += Bseg
        del endpoints[A]
        del endpoints[B]
        endpoints[Aseg[0]] = endpoints[Aseg[-1]] = Aseg
        return Aseg

    def tsp_tour_length(graph, tour):
        """ The TSP tour length WITH return to the starting point."""
        return sum(graph.links[tour[i - 1]][tour[i]] for i in range(len(tour)))
    
    def improve_tour(graph, tour):
        if not tour:
            raise ValueError("No tour to improve?")

        while True:
            improvements = {reverse_segment_if_improvement(graph, tour, i, j)
                            for (i, j) in subsegments(len(tour))}
            if improvements == {None} or len(improvements) == 0:
                return tour

    @lru_cache()
    def subsegments(N):
        """ Return (i, j) pairs denoting tour[i:j] subsegments of a tour of length N."""
        return [(i, i + length) for length in reversed(range(2, N))
                for i in reversed(range(N - length + 1))]
    
    def reverse_segment_if_improvement(graph, tour, i, j):
        """If reversing tour[i:j] would make the tour shorter, then do it."""
        # Given tour [...A,B...C,D...], consider reversing B...C to get [...A,C...B,D...]
        A, B, C, D = tour[i - 1], tour[i], tour[j - 1], tour[j % len(tour)]
        # Are old links (AB + CD) longer than new ones (AC + BD)? If so, reverse segment.
        if graph.links[A][B] + graph.links[C][D] > graph.links[A][C] + graph.links[B][D]:
            tour[i:j] = reversed(tour[i:j])
            return True

    # The core TSP solver:
    if not isinstance(graph, Graph):
        raise ValueError("Expected {} not {}".format(Graph.__class__.__name__, type(graph)))

    # 1. create a path using greedy algorithm (picks nearest peer)
    new_segment = []
    endpoints = {n: [n] for n in graph.nodes}
    L = shortest_links_first(graph)
    for _, a, b in L:
        if a in endpoints and b in endpoints and endpoints[a] != endpoints[b]:
            new_segment = join_endpoints(endpoints, a, b)
            if len(new_segment) == len(graph.nodes):
                break  # return new_segment
    assert len(new_segment) == len(graph.nodes)
    first_tour = new_segment[:]
    first_path_length = tsp_tour_length(graph, first_tour)

    # 2. run improvement on the created path.
    improved_tour = improve_tour(graph, new_segment)
    assert set(graph.nodes) == set(improved_tour)

    second_path_length = tsp_tour_length(graph, improved_tour)

    assert first_path_length >= second_path_length, "first path was better than improved tour?! {} {}".format(
        first_path_length, second_path_length
    )

    return second_path_length, improved_tour
