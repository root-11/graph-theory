from itertools import chain
from collections import defaultdict, deque
from collections.abc import Iterable


class BasicGraph(object):
    """
    BasicGraph is the base graph that all methods use.
    For methods, please see the documentation on the
    individual functions, by importing them separately.
    """

    def __init__(self, from_dict=None, from_list=None):
        """
        :param from_dict: creates graph for dictionary {n1:{n2:d} ...
        :param from_list: creates graph from list of edges(n1,n2,d)
        """
        self._nodes = {}
        self._edges = defaultdict(dict)
        self._edge_count = 0
        self._reverse_edges = defaultdict(dict)
        self._in_degree = defaultdict(int)
        self._out_degree = defaultdict(int)

        if from_dict is not None:
            self.from_dict(from_dict)
        elif from_list is not None:
            self.from_list(from_list)

    def __str__(self):
        return f"{self.__class__.__name__}({len(self._nodes)} nodes, {self._edge_count} edges)"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self._nodes != other._nodes:
            return False
        if self._edge_count != other._edge_count:
            return False
        if self._edges != other._edges:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

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

    def copy(self):
        cls = type(self)
        g = cls()
        for n in self._nodes:
            g.add_node(n, obj=self._nodes[n])
        for s, e, d in self.edges():
            g.add_edge(s, e, d)
        return g

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

        if node1 in self._edges and node2 in self._edges[node1]:  # it's a value update.
            self._edges[node1][node2] = value
            self._reverse_edges[node2][node1] = value
        else:  # it's a new edge.
            self._edges[node1][node2] = value
            self._reverse_edges[node2][node1] = value
            self._out_degree[node1] += 1
            self._in_degree[node2] += 1
            self._edge_count += 1

        if bidirectional:
            self.add_edge(node2, node1, value, bidirectional=False)

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

    def reverse_edge(self, node2, node1, default=None):
        """retrieves the edge from node2 to node1"""
        try:
            return self._reverse_edges[node2][node1]
        except KeyError:
            return default

    def del_edge(self, node1, node2):
        """
        removes edge from node1 to node2
        :param node1: node
        :param node2: node
        """
        try:
            del self._edges[node1][node2]
        except KeyError:
            return
        if not self._edges[node1]:
            del self._edges[node1]

        del self._reverse_edges[node2][node1]
        if not self._reverse_edges[node2]:
            del self._reverse_edges[node2]

        self._out_degree[node1] -= 1
        self._in_degree[node2] -= 1
        self._edge_count -= 1

    def add_node(self, node_id, obj=None):
        """
        :param node_id: any hashable node.
        :param obj: any object that the node should refer to.
        PRO TIP: To retrieve the node obj use g.node(node_id)
        """
        if node_id in self._nodes:  # it's an object update.
            self._nodes[node_id] = obj
        else:
            self._nodes[node_id] = obj
            self._in_degree[node_id] = 0
            self._out_degree[node_id] = 0

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
        if node_id not in self._nodes:
            return

        # outgoing
        for n2, _ in self._edges[node_id].copy().items():
            self.del_edge(node_id, n2)

        # incoming
        for n1, _ in self._reverse_edges[node_id].copy().items():
            self.del_edge(n1, node_id)

        for _d in [
            self._edges,
            self._reverse_edges,
            self._nodes,
            self._in_degree,
            self._out_degree]:
            try:
                del _d[node_id]
            except KeyError:
                pass

    def nodes(self, from_node=None, to_node=None, in_degree=None, out_degree=None):
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
            if from_node in self._edges:
                return list(self._edges[from_node])
            return []

        if to_node is not None:
            return list(self._reverse_edges[to_node])

        if in_degree is not None:
            if not isinstance(in_degree, int) or in_degree < 0:
                raise ValueError("in_degree must be int >= 0")
            return [n for n, cnt in self._in_degree.items() if cnt == in_degree]

        if out_degree is not None:
            if not isinstance(out_degree, int) or out_degree < 0:
                raise ValueError("out_degree must be int >= 0")
            return [n for n, cnt in self._out_degree.items() if cnt == out_degree]

    def edges(self, path=None, from_node=None, to_node=None):
        """
        :param path (optional) list of nodes for which the edges are wanted.
        :param from_node (optional) for which outgoing edges are returned.
        :param to_node (optional) for which incoming edges are returned.
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

            return [(path[ix], path[ix + 1], self._edges[path[ix]][path[ix + 1]]) for ix in range(len(path) - 1)]

        if from_node:
            if from_node in self._edges:
                return [(from_node, n2, cost) for n2, cost in self._edges[from_node].items()]
            else:
                return []

        if to_node:
            if to_node in self._reverse_edges:
                return [(n1, to_node, value) for n1, value in self._reverse_edges[to_node].items()]
            else:
                return []

        return [(n1, n2, out[n2]) for n1, out in self._edges.items() for n2 in out]

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
        if not isinstance(dictionary, dict):
            raise TypeError(f"expected dict, not {type(dictionary)}")
        for n1, e in dictionary.items():
            if not e:
                self.add_node(n1)
            else:
                for n2, v in e.items():
                    self.add_edge(n1, n2, v)

    def to_dict(self):
        """creates a nested dictionary from the graph.
        :return dict d[n1][n2] = distance
        """
        d = {n: {} for n in self.nodes()}
        for n1, n2, dist in self.edges():
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
            (11,)      # node with no links.
        ]
        """
        if not isinstance(links, Iterable):
            raise TypeError(f"Expected iterable, not {type(links)}")
        for item in links:
            assert isinstance(item, (list, tuple))
            if len(item) > 1:
                self.add_edge(*item)
            else:
                self.add_node(item[0])

    def to_list(self):
        """returns list of edges and nodes."""
        return self.edges() + [(i,) for i in self.nodes()]

    def is_connected(self, n1, n2):
        """helper determining if two nodes are connected using BFS."""
        if n1 in self._edges:
            q = deque([n1])
            visited = set()
            while q:
                n = q.popleft()
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

    def in_degree(self, node):
        """returns the number of edges incoming on a node"""
        return self._in_degree[node]

    def out_degree(self, node):
        """returns the number of edges departing from a node"""
        return self._out_degree[node]

    def distance(self, nodes, return_to_start=False):
        length = sum(self.edge(nodes[i - 1], nodes[i]) for i in range(len(nodes))) 
        if return_to_start:
            length += self.edge(nodes[-1], nodes[0])
        return length
        # return sum(self.edge(n1, n2, default=float("inf")) for n1, n2 in zip(nodes[:-1], nodes[1:]))


def subgraph(graph, nodes):
    """Creates a subgraph as a copy from the graph
    :param graph: class Graph
    :param nodes: set or list of nodes
    :return: new instance of Graph.
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected BasicGraph, Graph or Graph3D, not {type(graph)}")
    if not isinstance(nodes, (set, list)):
        raise TypeError(f"expected nodes as a set or a list, not {type(nodes)}")

    node_set = set(nodes)
    cls = type(graph)
    g = cls()
    for n1 in nodes:
        obj = graph.node(n1)
        g.add_node(n1, obj)
        for n2 in graph.nodes(from_node=n1):
            if n2 in node_set:
                g.add_edge(n1, n2, graph.edge(n1, n2))
    return g


def is_subgraph(graph1, graph2):
    """
    Checks if graph1 is subgraph in graph2
    :param graph1: instance of Graph
    :param graph2: instance of Graph
    :return: boolean
    """
    if not isinstance(graph1, BasicGraph):
        raise TypeError(f"Expected BasicGraph, Graph or Graph3D, not {type(graph1)}")
    if not isinstance(graph2, BasicGraph):
        raise TypeError(f"Expected BasicGraph, Graph or Graph3D, not {type(graph2)}")

    if not set(graph1.nodes()).issubset(set(graph2.nodes())):
        return False
    if not set(graph1.edges()).issubset(set(graph2.edges())):
        return False
    return True


def same_path(path1, path2):
    """Compares two paths to verify whether they're the same despite being offset.
    Very useful when comparing results from TSP as solutions may be rotations of
    the same path.
    :param path1: list of nodes.
    :param path2: list of nodes.
    :return: boolean.
    """
    if not isinstance(path1, (list, set, tuple)):
        raise TypeError(f"Expected path1 as Iterable, not {type(path1)}")
    if not isinstance(path2, (list, set, tuple)):
        raise TypeError(f"Expected path2 as Iterable, not {type(path2)}")

    if path1 is path2:  # use id system to avoid work.
        return True
    if len(path1) != len(path2) or set(path1) != set(path2):
        return False

    starts = (ix for ix, n2 in enumerate(path2) if path1[0] == n2)
    return any(all(a == b for a, b in zip(path1, chain(path2[start:], path2[:start]))) for start in starts)


def has_path(graph, path):
    """checks if path exists is graph
    :param graph: instance of Graph
    :param path: list of nodes
    :return: boolean
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected BasicGraph, Graph or Graph3D, not {type(graph)}")
    if not isinstance(path, (list, tuple)):
        raise TypeError(f"Expected list or tuple, not {type(path)}")
    v1 = path[0]
    for v2 in path[1:]:
        if graph.edge(v1, v2) is None:
            return False
        else:
            v1 = v2
    return True


def network_size(graph, n1, degrees_of_separation=None):
    """Determines the nodes within the range given by
    a degree of separation
    :param graph: Graph
    :param n1: start node
    :param degrees_of_separation: integer
    :return: set of nodes within given range
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected BasicGraph, Graph or Graph3D, not {type(graph)}")
    if n1 not in graph:
        raise ValueError(f"{n1} not in graph")

    if degrees_of_separation is not None:
        if not isinstance(degrees_of_separation, int):
            raise TypeError(f"Expected degrees_of_separation to be integer, not {type(degrees_of_separation)}")

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


def components(graph):
    """Determines the components of the graph
    :param graph: instance of class Graph
    :return: list of sets of nodes. Each set is a component.
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected BasicGraph, Graph or Graph3D, not {type(graph)}")

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
