__all__ = ['Graph']
from itertools import count

from graph.graphs import BasicGraph
from graph.topology import subgraph, is_subgraph, is_partite, same, has_path, has_cycles, components
from graph.flow_problem import maximum_flow
from graph.search import (
    shortest_path,
    breadth_first_search,
    depth_first_search,
    tsp,
    shortest_tree_all_pairs,
    all_paths,
    distance
)
from graph.transform import adjacency_matrix, all_pairs_shortest_paths


class Graph(BasicGraph):
    """
    Graph is the base graph that all methods use.

    For methods, please see the documentation on the
    individual functions, by importing them separately.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __copy__(self):
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


class Graph2D(BasicGraph):
    nid = count()
    """ a graph where all (x,y)-positions are unique. """
    def __init__(self, from_xy_dict=None, from_xy_list=None):
        super().__init__()
        self.xys = {}
        if from_xy_dict is not None:
            self.from_xy_dict(dictionary=from_xy_dict)
        if from_xy_list is not None:
            self.from_xy_list(links=from_xy_list)

    def __copy__(self):
        g = Graph2D(from_xy_dict=self.to_xy_dict())
        return g

    def add_xy_edge(self, x1, y1, x2, y2, value, bidirectional=False):
        nid_1 = self.xys.get((x1, y1), None)
        if nid_1 is None:
            nid_1 = self.add_xy_node(x1, y1)
        nid_2 = self.xys.get((x2, y2), None)
        if nid_2 is None:
            nid_2 = self.add_xy_node(x2, y2)

        self.add_edge(nid_1, nid_2, value=value, bidirectional=bidirectional)

    def add_xy_node(self, x, y, node_id=None):
        """ Adds node from xy coordinates """
        obj = (x, y)
        if obj in self.xys:
            raise ValueError("({},{}) already exists".format(x, y))
        if node_id is None:
            node_id = next(Graph2D.nid)
        self.add_node(node_id, obj)
        self.xys[obj] = node_id
        return node_id

    def node_id(self, x, y):
        return self.xys.get((x, y), None)

    def from_xy_dict(self, dictionary):
        """
        Updates the graph from dictionary
        :param dictionary

        d = {(1, 2): {(1, 7): 5,
                      (6, 2): 5},
             (1, 7): {(4, 4): 4,
                      (6, 2): 9},
             (6, 2): {(1, 7): 9,
                      (4, 4): 5,
                      (8, 5): 5.6},
             (4, 4): {(8, 5): 4},
             (8, 5): {(1, 2): 7,
                      (4, 4): 6}}
        G = Graph2D(from_dict=d)
        """
        def check(xy):
            assert isinstance(xy, tuple)
            assert len(xy) == 2
            assert all((isinstance(a, (float, int)) for a in xy))

        for k, v in dictionary.items():
            check(k)
            if k not in self.xys:
                x1, y1 = k
                nid_1 = self.add_xy_node(x1, y1)
            else:
                nid_1 = self.xys[k]
            for k2, v2 in v.items():
                check(k2)
                assert isinstance(v2, (int, float))
                if k2 not in self.xys:
                    x2, y2 = k2
                    nid_2 = self.add_xy_node(x2, y2)
                else:
                    nid_2 = self.xys[k2]
                self.add_edge(nid_1, nid_2, value=v2)

    def to_xy_dict(self):
        """ creates a nested dictionary from the graph.
        :return dict d[xy_1][xy_2] = distance
        """
        d = {}
        for n1, n2, dist in self.edges():
            xy1 = self._nodes[n1]
            xy2 = self._nodes[n2]
            if xy1 not in d:
                d[xy1] = {}
            d[xy1][xy2] = dist

        for n in self.nodes():
            xy = self._nodes[n]
            if xy not in d:
                d[xy] = {}
        return d

    def from_xy_list(self, links):
        """
        updates the graph from a list of xy coordinate pairs
        :param links:

        links = [
            ((1,2),(3,4),5),
            ((1,2),(4,3),5),
            ((2,4),(7,1),7),
            ((3,3),(7,1),7),
            ((4,4),)
        ]
        """
        assert isinstance(links, list)
        for item in links:
            assert isinstance(item, tuple)
            if len(item) == 3:
                self.add_xy_edge(*item)
            else:
                self.add_xy_node(*item[0])

    def to_xy_list(self):
        """ returns list of edges and nodes."""
        def xy(nid):
            return self.node(nid)

        return [(xy(s), xy(e), d) for s, e, d in self.edges()] + [(xy(i),) for i in self.nodes()]

    @staticmethod
    def xy_distance(x1, y1, x2, y2):
        """ returns the distance between to xy coordinates"""
        return ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) ** (1 / 2)

    def n_nearest_neighbours(self, x, y, n=1):
        """ returns the node id of the `n` nearest neighbours. """
        dist = self.xy_distance
        d = [(dist(x, y, x1, y1), n) for (x1, y1), n in self.xys.items()]
        d.sort()
        if d:
            return [b for a, b in d][:n]
        return None

    def plot(self, nodes=True, links=True):
        """ plots nodes and links using matplotlib3"""
        raise NotImplementedError  # TODO see test_graph\test_random_graph_3 for plotting.


class Graph3D(BasicGraph):
    def __init__(self):
        super().__init__()
        # TODO