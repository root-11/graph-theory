from functools import lru_cache
from itertools import combinations, chain
from collections import defaultdict
from heapq import heappop, heappush

__all__ = ['Graph',
           'subgraph',
           'shortest_path', 'distance', 'same',
           'tsp', 'all_paths']


class Graph(object):
    """
    Graph is the base graph that all methods use.

    For methods, please see the documentation on the
    individual functions, by importing them separately.

    """

    def __init__(self, from_dict=None, from_list=None):
        """
        :param from_dict: creates graph for dictionary {n1:{n2:d} ...
        :param links: creates graph from list of edges(n1,n2,d)
        """
        self._nodes = {}
        self._links = {}
        self._max_length = 0

        if from_dict is not None:
            self.from_dict(from_dict)
        elif from_list is not None:
            self.from_list(from_list)

    def __getitem__(self, item):
        out = None
        if item in self._links:
            out = self._links.__getitem__(item)
        elif item in self._nodes:
            out = self._nodes.__getitem__(item)
        if out is None:
            raise KeyError
        return out

    def __setitem__(self, key, value):
        raise ValueError("Use add_edge(node1, node2, value)")

    def __delitem__(self, key):
        self._links.__delitem__(key)

    def __contains__(self, item):
        return self._links.__contains__(item)

    def __len__(self):
        return len(self._nodes)

    def __copy__(self):
        return Graph(from_list=self.edges())

    def nodes(self):
        """
        :return: list of node ids.
        PRO TIP: To retrieve the node obj use g[node id]
        """
        return list(self._nodes.keys())

    def edges(self, path=None):
        """
        :param: along_path (optional) list of nodes.
        :return: list of edges (n1, n2, value)
        """
        if path:
            L = []
            for ix in range(len(path) - 1):
                n1, n2 = path[ix], path[ix + 1]
                L.append((n1, n2, self._links[n1][n2]))
        else:
            L = [(n1, n2, self._links[n1][n2]) for n1 in self._links for n2 in self._links[n1]]
        return L

    def add_node(self, node_id, obj=None):
        """
        :param node_id: any hashable node.
        :param obj: any object that the node should refer to.

        PRO TIP:
        If you want to hold additional values on your node, then define
        you class with a __hash__() method. See CustomNode as example.
        """
        self._nodes[node_id] = obj

    def add_edge(self, node1, node2, value=1, bidirectional=False):
        """
        :param node1: hashable node
        :param node2: hashable node
        :param value: numeric value (int or float)
        :param bidirectional: boolean.
        """
        if isinstance(value, (dict, list, tuple)):
            raise ValueError("value cannot be {}".format(type(value)))
        self.add_node(node1)
        self.add_node(node2)
        if node1 not in self._links:
            self._links[node1] = {}
        if node2 not in self._links:
            self._links[node2] = {}
        self._links[node1][node2] = value
        if value > self._max_length:
            self._max_length = value
        if bidirectional:
            self._links[node2][node1] = value

    def from_dict(self, dictionary):
        """
        Updates the graph from dictionary
        :param dictionary:

        d = {1: {2: 10, 3: 5},
             2: {4: 1, 3: 2},
             3: {2: 3, 4: 9, 5: 2},
             4: {5: 4},
             5: {1: 7, 4: 6}}
        G = from_dict(d)

        :return: None
        """
        for n1 in dictionary:
            for n2 in dictionary[n1]:
                self.add_edge(n1, n2, dictionary[n1][n2])

    def to_dict(self):
        """ creates a nested dictionary from the graph.
        :return dict d[n1][n2] = distance
        """
        d = {}
        for n1, n2, dist in self.edges():
            if n1 not in d:
                d[n1] = {}
            d[n1][n2] = dist
        return d

    def from_list(self, links):
        """
        updates the graph from a list of links.
        :param links: list

        links = [
            (1, 2, 18),
            (1, 3, 10),
            (2, 4, 7),
            (2, 5, 6),
            (3, 4, 2),
        ]
        """
        assert isinstance(links, list)
        for n1, n2, v in links:
            self.add_edge(n1, n2, v)

    def to_list(self):
        """ alias for self.edges()"""
        return self.edges()

    def shortest_path(self, start, end):
        """
        :param start: start node
        :param end: end node
        :return: distance, path as list
        """
        return shortest_path(graph=self, start=start, end=end)

    def breadth_first_search(self, start, end):
        """ Determines the path with fewest nodes.
        :param start: start node
        :param end: end nodes
        :return: nodes, path as list
        """
        return breadth_first_search(graph=self, start=start, end=end)

    def distance_from_path(self, path):
        """
        :param path: list of nodes
        :return: distance along the path.
        """
        return distance(graph=self, path=path)

    def maximum_flow(self, start, end):
        """ Determines the maximum flow of the graph between
        start and end.
        :param start: node (source)
        :param end: node (sink)
        :return: flow, graph of flow.
        """
        return maximum_flow(self, start, end)

    def solve_tsp(self):
        """ solves the traveling salesman problem for the graph
        (finds the shortest path through all nodes)
        :return: tour length (path+retrun to starting point),
                 path travelled.
        """
        return tsp(self)

    def subgraph_from_nodes(self, nodes):
        """
        constructs a copy of the graph containing only the
        listed nodes (and their links)
        :param nodes: list of nodes
        :return: class Graph
        """
        return subgraph(graph=self, nodes=nodes)

    def is_subgraph(self, other):
        """ Checks if self is a subgraph in other.
        :param other: instance of Graph
        :return: boolean
        """
        return is_subgraph(self, other)

    @staticmethod
    def same_path(p1, p2):
        """ compares two paths to determine if they're the same, despite
        being in different order.
        :param p1: list of nodes
        :param p2: list of nodes
        :return: boolean
        """
        return same(p1, p2)

    def adjacency_matrix(self):
        """
        :return:
        """
        return adjacency_matrix(graph=self)

    def all_pairs_shortest_paths(self):
        """
        :return:
        """
        return all_pairs_shortest_paths(graph=self)

    def shortest_tree_all_pairs(self):
        """
        :return:
        """
        return shortest_tree_all_pairs(graph=self)

    def has_path(self, path):
        """
        :param path: list of nodes
        :return: boolean, if the path is in G.
        """
        return has_path(graph=self, path=path)

    def all_paths(self, start, end):
        """
        finds all paths from start to end
        :param start: node
        :param end: node
        :return: list of paths
        """
        return all_paths(graph=self, start=start, end=end)


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
                next_node = cost + 1
                if prev is None or next_node < prev:
                    mins[v2] = next_node
                    heappush(q, (next_node, v2, path))
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
    for idx in range(len(path) - 1):
        n1, n2 = path[idx], path[idx + 1]
        d += graph[n1][n2]
    return d


def subgraph(graph, nodes):
    """ Creates a subgraph as a copy from the graph
    :param graph: class Graph
    :param nodes: list of nodes
    :return: new instance of Graph.
    """
    assert isinstance(graph, Graph)
    assert isinstance(nodes, list)
    G = Graph()
    for n1 in nodes:
        G.add_node(n1)
        for n2 in graph[n1]:
            G.add_edge(n1, n2, graph[n1][n2])
    return G


def is_subgraph(graph1, graph2):
    """
    Checks is graph1 is subgraph in graph2
    :param graph1: instance of Graph
    :param graph2: instance of Graph
    :return: boolean
    """
    assert isinstance(graph1, Graph)
    assert isinstance(graph2, Graph)
    if not set(graph1.nodes()).issubset(set(graph2.nodes())):
        return False
    if not set(graph1.edges()).issubset(set(graph2.edges())):
        return False
    return True


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
        c = combinations(graph.nodes(), 2)
        distances = [(graph[a][b], a, b) for a, b in c]
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
        return sum(graph[tour[i - 1]][tour[i]] for i in range(len(tour)))

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
        if graph[a][b] + graph[c][d] > graph[a][c] + graph[b][d]:
            tour[i:j] = reversed(tour[i:j])
            return True

    # The core TSP solver:
    assert isinstance(graph, Graph), type(graph)

    # 1. create a path using greedy algorithm (picks nearest peer)
    new_segment = []
    endpoints = {n: [n] for n in graph.nodes()}
    L = shortest_links_first(graph)
    for _, a, b in L:
        if a in endpoints and b in endpoints and endpoints[a] != endpoints[b]:
            new_segment = join_endpoints(endpoints, a, b)
            if len(new_segment) == len(graph):
                break  # return new_segment
    assert len(new_segment) == len(graph)
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


def adjacency_matrix(graph):
    """
    :param graph:
    :return: dictionary

    Converts directed graph to an adjacency matrix.
    Note: The distance from a node to itself is 0 and distance from a node to
    an unconnected node is defined to be infinite. This does not mean that there
    is no path from a node to another via other nodes.
        g = {1: {2: 3, 3: 8, 5: -4},
             2: {4: 1, 5: 7},
             3: {2: 4},
             4: {1: 2, 3: -5},
             5: {4: 6}}
        adj(g)
        {1: {1: 0, 2: 3, 3: 8, 4: inf, 5: -4},
         2: {1: inf, 2: 0, 3: inf, 4: 1, 5: 7},
         3: {1: inf, 2: 4, 3: 0, 4: inf, 5: inf},
         4: {1: 2, 2: inf, 3: -5, 4: 0, 5: inf},
         5: {1: inf, 2: inf, 3: inf, 4: 6, 5: 0}}
    """
    assert isinstance(graph, Graph)
    return {v1: {v2: 0 if v1 == v2 else graph[v1].get(v2, float('inf')) for v2 in graph.nodes()} for v1 in
            graph.nodes()}


def all_pairs_shortest_paths(graph):
    """
    Find the cost of the shortest path between every pair of vertices in a
    weighted graph. Uses the Floyd-Warshall algorithm.

    inf = float('inf')
    g = {0: {0: 0,   1: 1,   2: 4},
         1: {0: inf, 1: 0,   2: 2},
         2: {0: inf, 1: inf, 2: 0}}
    fw(g) #
    {0: {0: 0,   1: 1,   2: 3},
    1: {0: inf, 1: 0,   2: 2},
    2: {0: inf, 1: inf, 2: 0}}
    h = {1: {2: 3, 3: 8, 5: -4},
         2: {4: 1, 5: 7},
         3: {2: 4},
         4: {1: 2, 3: -5},
         5: {4: 6}}
    fw(adj(h)) #
        {1: {1: 0, 2: 1, 3: -3, 4: 2, 5: -4},
         2: {1: 3, 2: 0, 3: -4, 4: 1, 5: -1},
         3: {1: 7, 2: 4, 3: 0, 4: 5, 5: 3},
         4: {1: 2, 2: -1, 3: -5, 4: 0, 5: -2},
         5: {1: 8, 2: 5, 3: 1, 4: 6, 5: 0}}
    """
    g = graph.adjacency_matrix()
    assert isinstance(g, dict)
    vertices = g.keys()

    for v2 in vertices:
        g = {v1: {v3: min(g[v1][v3], g[v1][v2] + g[v2][v3])
                  for v3 in vertices}
             for v1 in vertices}
    return g


def shortest_tree_all_pairs(graph):
    """
       'minimize the longest distance between any pair'

    Note: This algorithm is not shortest path as it jumps
    to a new branch when it has exhausted a branch in the tree.
    :return: path
    """
    assert isinstance(graph, Graph)
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
        else:
            print("node {} is isolated, skipping...".format(start_node))  # it's an island.

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


def has_path(graph, path):
    """ checks if path exists is graph
    :param graph: instance of Graph
    :param path: list of nodes
    :return: boolean
    """
    assert isinstance(graph, Graph)
    assert isinstance(path, list)
    v1 = path[0]
    for v2 in path[1:]:
        try:
            _ = graph[v1][v2]
            v1 = v2
        except KeyError:
            return False
    return True


def all_paths(graph, start, end):
    """ Returns all paths from start to end.
    :param graph: instance of Graph
    :param start: node
    :param end: node
    :return: list of paths
    """
    assert isinstance(graph, Graph)
    options = set(graph.nodes()) - {start, end}

    # powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(options)
    L = []
    for combination in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)):
        path = [start] + list(combination) + [end]
        if has_path(graph, path):
            L.append(path)
    return L


def maximum_flow(graph, start, end):
    """
    Returns the maximum flow graph
    :param graph: instance of Graph
    :param start: node
    :param end: node
    :return: flow, graph
    """
    assert isinstance(graph, Graph)
    inflow = sum(graph[start][i] for i in graph[start])
    outflow = sum(d for s, e, d in graph.edges() if e == end)
    unassigned_flow = min(inflow, outflow)  # search in excess of this 'flow' is a waste of time.
    total_flow = 0

    # below:
    # I reviewed a number of algorithms, such as Ford-fulkerson algorithm,
    # Edmonson-Karp and Dinic, but I didn't like them due to their naive usage
    # of BFS, which leads to a lot of node visits.
    #
    # I therefore choose to invert the capacities of the graph so that the
    # capacity any G[u][v] = c becomes 1/c in G_inverted.
    # This allows me to use the shortest path method to find the path with
    # most capacity in the first attempt, resulting in a significant reduction
    # of unassigned flow.
    #
    # By updating G_inverted, with the residual capacity, I can keep using the
    # shortest path, until the capacity is zero, whereby I remove the links
    # When the shortest path method returns 'No path' or when unassigned flow
    # is zero, I exit the algorithm.

    edges = [(n1, n2, 1 / d) for n1, n2, d in graph.edges() if d > 0]
    inverted_graph = Graph(from_list=edges)  # create G_inverted.

    flow_graph = Graph()  # Create structure to record flows.

    while unassigned_flow:
        # 1. find the best path
        d, path = shortest_path(inverted_graph, start, end)
        if d == float('inf'):  # then there is no path, and we must exit.
            return total_flow, flow_graph
        # else: use the path and lookup the actual flow from the capacity graph.
        path_flow = min(d for s, e, d in graph.edges(path))

        # 2. update the unassigned flow.
        unassigned_flow -= path_flow
        total_flow += path_flow

        # 3. record the flows and update the inverted graph, so that it is
        #    ready for the next iteration.
        edges = graph.edges(path)
        for n1, n2, d in edges:

            # 3.a. recording:
            if n1 in flow_graph and n2 in flow_graph[n1]:
                flow_graph[n1][n2] += path_flow
            else:
                flow_graph.add_edge(n1, n2, path_flow)

            # 3.b. updating:
            # if there is capacity left: update with new 1/capacity
            # else: remove node, as we can't do 1/zero.
            new_capacity = graph[n1][n2] - path_flow
            if new_capacity > 0:
                inverted_graph[n1][n2] = 1/new_capacity
            else:
                del inverted_graph[n1][n2]
    return total_flow, flow_graph



