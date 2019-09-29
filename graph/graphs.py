

class BasicGraph(object):
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

    def __copy__(self):
        if not self.__class__.__name__ == BasicGraph.__name__:
            raise NotImplementedError("subclasses must implement this method.")
        g = BasicGraph()
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
            # return [n2 for n1, n2, d in self.edges(from_node=from_node)]
            if self._edges.get(from_node, None) is not None:
                return [n2 for n2, v in self._edges[from_node].items()]
            return []

        if to_node is not None:
            return [n1 for n1, n2, d in self.edges() if n2 == to_node]

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
                    for ix in range(len(path ) -1)]

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

