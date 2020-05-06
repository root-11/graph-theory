from collections import defaultdict
from functools import lru_cache
from heapq import heappop, heappush
from itertools import combinations

from graph.visuals import plot_3d

__description__ = """
The graph-theory library is organised in the following way for clarity of structure:

1. BasicGraph (class) - with general methods for all subclasses.
2. All methods for class Graph in same order as on Graph.
3. Graph (class)
4. Graph3D (class) 

"""


class BasicGraph(object):
    """
    BasicGraph is the base graph that all methods use.

    For methods, please see the documentation on the
    individual functions, by importing them separately.
    """

    def __init__(self, from_dict=None, from_list=None):
        """
        :param from_dict: creates graph for dictionary {n1:{n2:d} ...
        :param links: creates graph from list of edges(n1,n2,d)
        """
        self._nodes = {}
        self._edges = {}
        self._max_edge_value = 0

        if from_dict is not None:
            self.from_dict(from_dict)
        elif from_list is not None:
            self.from_list(from_list)

    def __getitem__(self, item):
        raise ValueError("Use g.node(n1) or g.edge(n1,n2)")

    def __setitem__(self, key, value):
        raise ValueError("Use add_edge(node1, node2, value)")

    def __delitem__(self, key):
        raise ValueError("Use del_edge(node1, node2)")

    def __contains__(self, item):
        """
        :returns bool: True if node in Graph.
        """
        return item in self._nodes

    def __len__(self):
        raise ValueError("Use len(g.nodes()) or len(g.edges())")

    def add_edge(self, node1, node2, value=1, bidirectional=False):
        """
        :param node1: hashable node
        :param node2: hashable node
        :param value: numeric value (int or float)
        :param bidirectional: boolean.
        """
        if isinstance(value, (dict, list, tuple)):
            raise ValueError("value cannot be {}".format(type(value)))
        if node1 not in self._nodes:
            self.add_node(node1)
        if node2 not in self._nodes:
            self.add_node(node2)

        if node1 not in self._edges:
            self._edges[node1] = {}
        if node2 not in self._edges:
            self._edges[node2] = {}
        self._edges[node1][node2] = value
        if value > self._max_edge_value:
            self._max_edge_value = value
        if bidirectional:
            self._edges[node2][node1] = value

    def edge(self, node1, node2, default=None):
        """Retrieves the edge (node1, node2)

        Alias for g[node1][node2]

        :param node1: node id
        :param node2: node id
        :param default: returned value if edge doesn't exist.
        :return: edge(node1,node2)
        """
        try:
            return self._edges[node1][node2]
        except KeyError:
            return default

    def del_edge(self, node1, node2):
        """
        removes edge from node1 to node2
        :param node1: node
        :param node2: node
        """
        del self._edges[node1][node2]

    def add_node(self, node_id, obj=None):
        """
        :param node_id: any hashable node.
        :param obj: any object that the node should refer to.

        PRO TIP: To retrieve the node obj use g.node(node_id)

        """
        self._nodes[node_id] = obj

    def node(self, node_id):
        """
        Retrieves the node object

        :param node_id: id of node in graph.
        :return: node object
        """
        return self._nodes.get(node_id, None)

    def del_node(self, node_id):
        """
        Deletes the node and all its connections.
        :param node_id: node_id
        :return: None
        """
        try:
            del self._nodes[node_id]
        except KeyError:
            pass
        try:
            del self._edges[node_id]
        except KeyError:
            pass
        in_links = [n1 for n1, n2, d in self.edges() if n2 == node_id]
        for inlink in in_links:
            del self._edges[inlink][node_id]
        return None

    def nodes(self,
              from_node=None, to_node=None,
              in_degree=None, out_degree=None):
        """
        :param from_node (optional) return nodes with edges from 'from_node'
        :param to_node (optional) returns nodes with edges into 'to_node'
        :param in_degree (optional) returns nodes with in_degree=N
        :param out_degree (optional) returns nodes with out_degree=N

        :return list of node ids.
        """
        inputs = sum([1 for i in (from_node, to_node, in_degree, out_degree) if i is not None])
        if inputs > 1:
            m = []
            a = (from_node, to_node, in_degree, out_degree)
            b = ("from_node", "to_node", "in_degree", "out_degree")
            for i in zip(a, b):
                if i is not None:
                    m.append("{}={}".format(b, a))
            raise ValueError("nodes({}) has too many inputs. Pick one.".format(m))

        if inputs == 0:
            return list(self._nodes.keys())

        if from_node is not None:
            if self._edges.get(from_node, None) is not None:
                return [n2 for n2 in self._edges[from_node]]
            return []

        if to_node is not None:
            return [n1 for n1 in self._edges for n2 in self._edges[n1] if n2 == to_node]

        if in_degree is not None:
            if not isinstance(in_degree, int) or in_degree < 0:
                raise ValueError("in_degree must be int >= 0")

            rev = {n: set() for n in self._nodes}
            for n1, n2, d in self.edges():
                rev[n2].add(n1)
            return [n for n, n_set in rev.items() if len(n_set) == in_degree]

        if out_degree is not None:
            if not isinstance(out_degree, int) or out_degree < 0:
                raise ValueError("out_degree must be int >= 0")

            rev = {n: set() for n in self._nodes}
            for n1, n2, d in self.edges():
                rev[n1].add(n2)
            return [n for n, n_set in rev.items() if len(n_set) == out_degree]

    def edges(self, path=None, from_node=None, to_node=None):
        """
        :param path (optional) list of nodes for which the edges are wanted.
        :param from_node (optional) for which outgoing edges are returned.
        :param to_node (optiona) for which incoming edges are returned.
        :return list of edges (n1, n2, value)
        """
        inputs = sum([1 for i in (from_node, to_node, path) if i is not None])
        if inputs > 1:
            m = []
            a = (path, from_node, to_node)
            b = ("path", "from_node", "to_node")
            for i in zip(a, b):
                if i is not None:
                    m.append("{}={}".format(b, a))
            raise ValueError("edges({}) has too many inputs. Pick one.".format(m))

        if path:
            if not isinstance(path, list):
                raise ValueError("expects a list")
            if len(path) < 2:
                raise ValueError("path of length 1 is not a path.")

            return [(path[ix], path[ix + 1], self._edges[path[ix]][path[ix + 1]])
                    for ix in range(len(path) - 1)]

        if from_node:
            if from_node in self._edges:
                return [(from_node, n2, self._edges[from_node][n2]) for n2 in self._edges[from_node]]
            else:
                return []

        if to_node:
            return [(n1, n2, self._edges[n1][n2])
                    for n1 in self._edges
                    for n2 in self._edges[n1]
                    if n2 == to_node]

        return [(n1, n2, self._edges[n1][n2]) for n1 in self._edges for n2 in self._edges[n1]]

    def from_dict(self, dictionary):
        """
        Updates the graph from dictionary
        :param dictionary:

        d = {1: {2: 10, 3: 5},
             2: {4: 1, 3: 2},
             3: {2: 3, 4: 9, 5: 2},
             4: {5: 4},
             5: {1: 7, 4: 6}}

        G = Graph(from_dict=d)

        :return: None
        """
        assert isinstance(dictionary, dict)
        for n1, e in dictionary.items():
            if not e:
                self.add_node(n1)
            else:
                for n2, v in e.items():
                    self.add_edge(n1, n2, v)

    def to_dict(self):
        """ creates a nested dictionary from the graph.
        :return dict d[n1][n2] = distance
        """
        d = {}
        for n1, n2, dist in self.edges():
            if n1 not in d:
                d[n1] = {}
            d[n1][n2] = dist

        for n in self.nodes():
            if n not in d:
                d[n] = {}
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
            (11,)      # node with no links.
        ]
        """
        assert isinstance(links, list)
        for item in links:
            assert isinstance(item, tuple)
            if len(item) == 3:
                self.add_edge(*item)
            else:
                self.add_node(item[0])

    def to_list(self):
        """ returns list of edges and nodes."""
        return self.edges() + [(i,) for i in self.nodes()]

    @lru_cache(maxsize=128)
    def is_connected(self, n1, n2):
        """ helper determining if two nodes are connected using BFS. """
        q = [n1]
        visited = set()
        while q:
            n = q.pop(0)
            if n not in visited:
                visited.add(n)
            for c in self._edges[n]:
                if c == n2:
                    return True  # <-- Exit if connected.
                if c in visited:
                    continue
                else:
                    q.append(c)
        return False  # <-- Exit if not connected.


# Graph functions
# -----------------------------
def shortest_path(graph, start, end):
    """
    :param graph: class Graph
    :param start: start node
    :param end: end node
    :return: distance, path (as list),
             returns float('inf'), [] if no path exists.
    """
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
    if start not in graph:
        raise ValueError(f"{start} not in graph")
    if end not in graph:
        raise ValueError(f"{end} not in graph")

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


def depth_scan(graph, start, criteria):
    """ traverses the descendants of node `start` using callable `criteria` to determine
    whether to terminate search along each branch in `graph`.

    :param graph: class Graph
    :param start: start node
    :param criteria: function to terminate scan along a branch must return bool
    :return: set of nodes
    """
    if not callable(criteria):
        raise TypeError(f"Expected {criteria} to be callable")
    if start not in graph:
        raise ValueError(f"{start} not in graph")
    if not criteria(start):
        return set()

    q = [start]
    path = []
    visited = set()
    while q:
        n1 = q.pop()
        visited.add(n1)
        path.append(n1)
        for n2 in graph.nodes(from_node=n1):
            if n2 in visited:
                continue
            if not criteria(n2):
                visited.add(n2)
                continue
            q.append(n2)
            break
        else:
            path.remove(n1)
            while not q and path:
                for n2 in graph.nodes(from_node=path[-1]):
                    if n2 in visited:
                        continue
                    if not criteria(n2):
                        visited.add(n2)
                        continue
                    q.append(n2)
                    break
                else:
                    path = path[:-1]
    return visited


def distance(graph, path):
    """ Calculates the distance for the path in graph
    :param graph: class Graph
    :param path: list of nodes
    :return: distance
    """
    assert isinstance(path, list)
    cache = defaultdict(dict)
    path_length = 0
    for idx in range(len(path) - 1):
        n1, n2 = path[idx], path[idx + 1]

        # if the edge exists...
        d = graph.edge(n1, n2, default=None)
        if d:
            path_length += d
            continue

        # if we've seen the edge before...
        d = cache.get((n1, n2), None)
        if d:
            path_length += d
            continue

        # if there no alternative ... (search)
        d, _ = shortest_path(graph, n1, n2)
        if d == float('inf'):
            return float('inf')  # <-- Exit if there's no path.
        else:
            cache[(n1, n2)] = d
        path_length += d
    return path_length


def maximum_flow(graph, start, end):
    """
    Returns the maximum flow graph
    :param graph: instance of Graph
    :param start: node
    :param end: node
    :return: flow, graph
    """
    inflow = sum(d for s, e, d in graph.edges(from_node=start))
    outflow = sum(d for s, e, d in graph.edges(to_node=end))
    unassigned_flow = min(inflow, outflow)  # search in excess of this 'flow' is a waste of time.
    total_flow = 0
    # -----------------------------------------------------------------------
    # The algorithm
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
    #
    # Even on small graphs, this method is very efficient, despite the overhead
    # of using shortest path. For very large graphs, this method outperforms
    # all other algorithms by orders of magnitude.
    # -----------------------------------------------------------------------

    edges = [(n1, n2, 1 / d) for n1, n2, d in graph.edges() if d > 0]
    inverted_graph = BasicGraph(from_list=edges)  # create G_inverted.
    capacity_graph = BasicGraph()  # Create structure to record capacity left.
    flow_graph = BasicGraph()  # Create structure to record flows.

    while unassigned_flow:
        # 1. find the best path
        d, path = shortest_path(inverted_graph, start, end)
        if d == float('inf'):  # then there is no path, and we must exit.
            return total_flow, flow_graph
        # else: use the path and lookup the actual flow from the capacity graph.

        path_flow = min([min(d, capacity_graph.edge(s, e, default=float('inf')))
                         for s, e, d in graph.edges(path=path)])

        # 2. update the unassigned flow.
        unassigned_flow -= path_flow
        total_flow += path_flow

        # 3. record the flows and update the inverted graph, so that it is
        #    ready for the next iteration.
        edges = graph.edges(path)
        for n1, n2, d in edges:

            # 3.a. recording:
            v = flow_graph.edge(n1, n2, default=None)
            if v is None:
                flow_graph.add_edge(n1, n2, path_flow)
                c = graph.edge(n1, n2) - path_flow
            else:
                flow_graph.add_edge(n1, n2, value=v + path_flow)
                c = graph.edge(n1, n2) - (v + path_flow)
            capacity_graph.add_edge(n1, n2, c)

            # 3.b. updating:
            # if there is capacity left: update with new 1/capacity
            # else: remove node, as we can't do 1/zero.
            if c > 0:
                inverted_graph.add_edge(n1, n2, 1 / c)
            else:
                inverted_graph.del_edge(n1, n2)
    return total_flow, flow_graph


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


def subgraph(graph, nodes):
    """ Creates a subgraph as a copy from the graph
    :param graph: class Graph
    :param nodes: list of nodes
    :return: new instance of Graph.
    """
    assert isinstance(nodes, list)
    G = object.__new__(graph.__class__)
    assert isinstance(G, BasicGraph)
    G.__init__()
    for n1 in nodes:
        obj = graph.node(n1)
        G.add_node(n1, obj)
        for n2 in graph.nodes(from_node=n1):
            G.add_edge(n1, n2, graph.edge(n1, n2))
    return G


def is_subgraph(graph1, graph2):
    """
    Checks if graph1 is subgraph in graph2
    :param graph1: instance of Graph
    :param graph2: instance of Graph
    :return: boolean
    """
    assert isinstance(graph1, BasicGraph)
    assert isinstance(graph2, BasicGraph)
    if not set(graph1.nodes()).issubset(set(graph2.nodes())):
        return False
    if not set(graph1.edges()).issubset(set(graph2.edges())):
        return False
    return True


def is_partite(graph, n):
    """ Checks if graph is n-partite
    :param graph: class Graph
    :param n: int, number of partitions.
    :return: boolean and partitions as dict[colour] = set(nodes) or None.
    """
    assert isinstance(graph, BasicGraph)
    assert isinstance(n, int)
    colours_and_nodes = {i: set() for i in range(n)}
    nodes_and_colours = {}
    n1 = set(graph.nodes()).pop()
    q = [n1]
    visited = set()
    colour = 0
    while q:
        n1 = q.pop()
        visited.add(n1)

        if n1 in nodes_and_colours:
            colour = nodes_and_colours[n1]
        else:
            colours_and_nodes[colour].add(n1)
            nodes_and_colours[n1] = colour

        next_colour = (colour + 1) % n
        neighbours = graph.nodes(from_node=n1) + graph.nodes(to_node=n1)
        for n2 in neighbours:
            if n2 in nodes_and_colours:
                if nodes_and_colours[n2] == colour:
                    return False, None
                # else:  pass  # it already has a colour and there is no conflict.
            else:  # if n2 not in nodes_and_colours:
                colours_and_nodes[next_colour].add(n2)
                nodes_and_colours[n2] = next_colour
                continue
            if n2 not in visited:
                q.append(n2)

    return True, colours_and_nodes


def has_cycles(graph):
    """ Checks if graph has a cycle
    :param graph: instance of class Graph.
    :return: bool
    """
    for n1, n2, d in graph.edges():
        if n1 == n2:
            return True
        if graph.is_connected(n2, n1):
            return True
    return False


def components(graph):
    """ Determines the components of the graph
    :param graph: instance of class Graph
    :return: list of sets of nodes. Each set is a component.
    """
    assert isinstance(graph, BasicGraph)
    nodes = set(graph.nodes())
    sets_of_components = []
    while nodes:
        new_component = set()
        sets_of_components.append(new_component)
        n = nodes.pop()  # select random node
        new_component.add(n)  # add it to the new component.

        new_nodes = set(graph.nodes(from_node=n))
        new_nodes.update(set(graph.nodes(to_node=n)))
        while new_nodes:
            n = new_nodes.pop()
            new_component.add(n)
            new_nodes.update(set(n for n in graph.nodes(from_node=n) if n not in new_component))
            new_nodes.update(set(n for n in graph.nodes(to_node=n) if n not in new_component))
        nodes = nodes - new_component
    return sets_of_components


def network_size(graph, n1, degrees_of_separation=None):
    """ Determines the nodes within the range given by
    a degree of separation
    :param graph: Graph
    :param n1: start node
    :param degrees_of_separation: integer
    :return: set of nodes within given range
    """
    assert isinstance(graph, BasicGraph)
    assert n1 in graph.nodes()
    if degrees_of_separation is not None:
        assert isinstance(degrees_of_separation, int)

    network = {n1}
    q = set(graph.nodes(from_node=n1))

    scan_depth = 1
    while True:
        if not q:  # then there's no network.
            break

        if degrees_of_separation is not None:
            if scan_depth > degrees_of_separation:
                break

        new_q = set()
        for peer in q:
            if peer in network:
                continue
            else:
                network.add(peer)
                new_peers = set(graph.nodes(from_node=peer)) - network
                new_q.update(new_peers)
        q = new_q
        scan_depth += 1
    return network


def phase_lines(graph):
    """ Determines the phase lines of a directed graph.
    :param graph: Graph
    :return: dictionary with node id : phase in cut.
    """
    phases = {n: 0 for n in graph.nodes()}
    sinks = {n: set() for n in phases}  # sinks[e] = {s1,s2}
    edges = {n: set() for n in phases}
    for s, e, d in graph.edges():
        sinks[e].add(s)
        edges[s].add(e)

    level = 0
    while sinks:
        sources = [e for e in sinks if not sinks[e]]  # these nodes have in_degree=0
        if not sources:
            raise AttributeError("The graph does not have any sinks.")
        for s in sources:
            phases[s] = level  # let's update the phase value
            del sinks[s]  # and let's remove their sink entry.
            # and remove their set item from the sinks dict
            for e in edges[s]:
                if e not in sinks:
                    continue
                sinks[e].discard(s)
                # but also check if their descendants are loops.
                for s2 in list(sinks[e]):
                    if graph.is_connected(e, s2):
                        sinks[e].discard(s2)
        level += 1

    return phases


def sources(graph, n):
    """ Determines the set of sources of 'node' in a DAG
    :param graph: Graph
    :return: set of nodes
    """
    nodes = {n}
    q = [n]
    while q:
        new = q.pop(0)
        for src in graph.nodes(to_node=new):
            if src not in nodes:
                nodes.add(src)
            if src not in q:
                q.append(src)
    nodes.remove(n)
    return nodes


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


def adjacency_matrix(graph):
    """Converts directed graph to an adjacency matrix.
    :param graph:
    :return: dictionary

    The distance from a node to itself is 0 and distance from a node to
    an unconnected node is defined to be infinite. This does not mean that there
    is no path from a node to another via other nodes.

    Example:
        g = Graph(from_dict=
            {1: {2: 3, 3: 8, 5: -4},
             2: {4: 1, 5: 7},
             3: {2: 4},
             4: {1: 2, 3: -5},
             5: {4: 6}})

        adjacency_matrix(g)
        {1: {1: 0, 2: 3, 3: 8, 4: inf, 5: -4},
         2: {1: inf, 2: 0, 3: inf, 4: 1, 5: 7},
         3: {1: inf, 2: 4, 3: 0, 4: inf, 5: inf},
         4: {1: 2, 2: inf, 3: -5, 4: 0, 5: inf},
         5: {1: inf, 2: inf, 3: inf, 4: 6, 5: 0}}
    """
    assert isinstance(graph, BasicGraph)
    return {v1: {v2: 0 if v1 == v2 else graph.edge(v1, v2, default=float('inf'))
                 for v2 in graph.nodes()}
            for v1 in graph.nodes()}


def all_pairs_shortest_paths(graph):
    """Find the cost of the shortest path between every pair of vertices in a
    weighted graph. Uses the Floyd-Warshall algorithm.

    Example:
        inf = float('inf')
        g = Graph(from_dict=(
            {0: {0: 0,   1: 1,   2: 4},
             1: {0: inf, 1: 0,   2: 2},
             2: {0: inf, 1: inf, 2: 0}})

        fw(g)
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


def minsum(graph):
    """ finds the mode(s) that have the smallest sum of distance to all other nodes. """
    assert isinstance(graph, Graph)
    adj_mat = graph.all_pairs_shortest_paths()
    for n in adj_mat:
        adj_mat[n] = sum(adj_mat[n].values())
    smallest = min(adj_mat.values())
    return [k for k, v in adj_mat.items() if v == smallest]


def minmax(graph):
    """ finds the node(s) with shortest distance to all other nodes. """
    assert isinstance(graph, Graph)
    adj_mat = graph.all_pairs_shortest_paths()
    for n in adj_mat:
        adj_mat[n] = max(adj_mat[n].values())
    smallest = min(adj_mat.values())
    return [k for k, v in adj_mat.items() if v == smallest]


def shortest_tree_all_pairs(graph):
    """
       'minimize the longest distance between any pair'

    Note: This algorithm is not shortest path as it jumps
    to a new branch when it has exhausted a branch in the tree.
    :return: path
    """
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


def has_path(graph, path):
    """ checks if path exists is graph
    :param graph: instance of Graph
    :param path: list of nodes
    :return: boolean
    """
    assert isinstance(graph, BasicGraph)
    assert isinstance(path, list)
    v1 = path[0]
    for v2 in path[1:]:
        if graph.edge(v1, v2) is None:
            return False
        else:
            v1 = v2
    return True


def all_paths(graph, start, end):
    """
    :param graph: instance of Graph
    :param start: node
    :param end: node
    :return: list of paths unique from start to end.
    """
    if start == end:
        raise ValueError("start is end")
    if not graph.is_connected(start, end):
        return []
    paths = [(start,)]
    q = [start]
    skip_list = set()
    while q:
        n1 = q.pop(0)
        if n1 == end:
            continue

        n2s = graph.nodes(from_node=n1)
        new_paths = [p for p in paths if p[-1] == n1]
        for n2 in n2s:
            if n2 in skip_list:
                continue
            n3s = graph.nodes(from_node=n2)
            if len(n3s) > 1 and graph.is_connected(n2, n1):
                # it's a fork and it's a part of a loop!
                # is the sequence n2,n3 already in the path?
                for n3 in n3s:
                    for path in new_paths:
                        a = [n2, n3]
                        if any(all(path[i+j] == a[j] for j in range(len(a))) for i in range(len(path))):
                            skip_list.add(n3)

            for path in new_paths:
                if path in paths:
                    paths.remove(path)

                new_path = path + (n2,)
                if new_path not in paths:
                    paths.append(new_path)

            if n2 not in q:
                q.append(n2)

    paths = [list(p) for p in paths if p[-1] == end]
    return paths


def degree_of_separation(graph, n1, n2):
    """ Calculates the degree of separation between 2 nodes."""
    assert n1 in graph.nodes()
    d, p = breadth_first_search(graph, n1, n2)
    return d


class Graph(BasicGraph):
    """
    Graph is the base graph that all methods use.

    For methods, please see the documentation on the
    individual functions, by importing them separately.

    """

    def __init__(self, from_dict=None, from_list=None):
        super().__init__(from_dict=from_dict, from_list=from_list)

    def copy(self):
        g = Graph()
        for n in self._nodes:
            g.add_node(n, obj=self._nodes[n])
        for s, e, d in self.edges():
            g.add_edge(s, e, d)
        return g

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

    def depth_first_search(self, start, end):
        """
        Finds a path from start to end using DFS.
        :param start: start node
        :param end: end node
        :return: path
        """
        return depth_first_search(graph=self, start=start, end=end)

    def depth_scan(self, start, criteria):
        """
        traverses the descendants of node `start` using callable `criteria` to determine
        whether to terminate search along each branch in `graph`.

        :param graph: class Graph
        :param start: start node
        :param criteria: function to terminate scan along a branch must return bool
        :return: set of nodes
        """
        return depth_scan(graph=self, start=start, criteria=criteria)

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

    def is_partite(self, n=2):
        """ Checks if self is n-partite
        :param n: int the number of partitions.
        :return: tuple: boolean, partitions as dict
                        (or None if graph isn't n-partite)
        """
        return is_partite(self, n)

    def has_cycles(self):
        """ Checks if the graph has a cycle
        :return: bool
        """
        return has_cycles(graph=self)

    def components(self):
        """ Determines the number of components
        :return: list of sets of nodes. Each set is a component.
        """
        return components(graph=self)

    def network_size(self, n1, degrees_of_separation=None):
        """ Determines the nodes within the range given by
        a degree of separation
        :param graph: Graph
        :param n1: start node
        :param degrees_of_separation: integer
        :return: set of nodes within given range
        """
        return network_size(self, n1, degrees_of_separation)

    def phase_lines(self):
        """ Determines the phase lines (cuts) of the graph
        :param: check_if_cyclic: bool: performs check to detect if graph is cyclic.
        :returns: dictionary with phase: nodes in phase
        """
        return phase_lines(self)

    def sources(self, n):
        """ Determines the DAG sources of node n """
        return sources(graph=self, n=n)

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
        Converts directed graph to an adjacency matrix.
        Note: The distance from a node to itself is 0 and distance from a node to
        an unconnected node is defined to be infinite. This does not mean that there
        is no path from a node to another via other nodes.
        :return: dict
        """
        return adjacency_matrix(graph=self)

    def minsum(self):
        """ Finds the mode(s) that have the smallest sum of distance to all other nodes.
        :return: list of nodes
        """
        return minsum(self)

    def minmax(self):
        """ Finds the node(s) with shortest distance to all other nodes.
        :return: list of nodes
        """
        return minmax(self)

    def all_pairs_shortest_paths(self):
        """
        Find the cost of the shortest path between every pair of vertices in a
        weighted graph. Uses the Floyd-Warshall algorithm.
        :return: dict {node 1: {node 2: distance}, ...}
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

    def degree_of_separation(self, n1, n2):
        """ determines the degree of separation between 2 nodes
        :param n1: node
        :param n2: node
        :return: degree
        """
        return degree_of_separation(self, n1, n2)


class Graph3D(Graph):
    """ a graph where all (x,y)-positions are unique. """

    def __init__(self, from_dict=None, from_list=None):
        super().__init__(from_dict=from_dict, from_list=from_list)

    def copy(self):
        g = Graph3D(from_dict=self.to_dict())
        return g

    # spatial only function
    # ---------------------
    @staticmethod
    def _check_tuples(n1):
        if not isinstance(n1, tuple):
            raise TypeError(f"expected tuple, not {type(n1)}")
        if len(n1) != 3:
            raise ValueError(f"expected tuple in the form as (x,y,z), got {n1}")
        if not all(isinstance(i, (float, int)) for i in n1):
            raise TypeError(f"expected all values to be integer or float, but got {n1}")

    @staticmethod
    def distance(n1, n2):
        """ returns the distance between to xyz tuples coordinates
        :param n1: (x,y,z)
        :param n2: (x,y,z)
        :return: float
        """
        Graph3D._check_tuples(n1)
        Graph3D._check_tuples(n2)
        (x1, y1, z1), (x2, y2, z2) = n1, n2
        a = abs(x2 - x1)
        b = abs(y2 - y1)
        c = abs(z2 - z1)
        return (a * a + b * b + c * c) ** (1 / 2)

    def add_edge(self, n1, n2, value=None, bidirectional=False):
        self._check_tuples(n1)
        self._check_tuples(n2)
        assert value is not None
        super().add_edge(n1, n2, value, bidirectional)

    def add_node(self, node_id, obj=None):
        self._check_tuples(node_id)
        super().add_node(node_id, obj)
        """
        :param node_id: any hashable node.
        :param obj: any object that the node should refer to.

        PRO TIP: To retrieve the node obj use g.node(node_id)
        """
        self._nodes[node_id] = obj

    def n_nearest_neighbours(self, node_id, n=1):
        """ returns the node id of the `n` nearest neighbours. """
        self._check_tuples(node_id)
        if not isinstance(n, int):
            raise TypeError(f"expected n to be integer, not {type(n)}")
        if n < 1:
            raise ValueError(f"expected n >= 1, not {n}")

        d = [(self.distance(n1=node_id, n2=n), n) for n in self.nodes() if n != node_id]
        d.sort()
        if d:
            return [b for a, b in d][:n]
        return None

    def plot(self, nodes=True, edges=True, rotation='xyz', maintain_aspect_ratio=False):
        """ plots nodes and links using matplotlib3
        :param nodes: bool: plots nodes
        :param edges: bool: plots edges
        :param rotation: str: set view point as one of [xyz,xzy,yxz,yzx,zxy,zyx]
        :param maintain_aspect_ratio: bool: rescales the chart to maintain aspect ratio.
        :return: None. Plots figure.
        """
        return plot_3d(self, nodes, edges, rotation, maintain_aspect_ratio)
