__all__ = ['Graph', 'Graph3D']

from graph.graphs import BasicGraph
from graph.spatial_graph import Graph3D
from graph.topology import (
    subgraph,
    is_subgraph,
    is_partite,
    same,
    has_path,
    has_cycles,
    components,
    network_size,
    phase_lines,
    sources
)
from graph.flow_problem import maximum_flow
from graph.search import (
    shortest_path,
    breadth_first_search,
    depth_first_search,
    tsp,
    shortest_tree_all_pairs,
    all_paths,
    distance,
    degree_of_separation
)
from graph.transform import adjacency_matrix, all_pairs_shortest_paths


class Graph(BasicGraph):
    """
    Graph is the base graph that all methods use.

    For methods, please see the documentation on the
    individual functions, by importing them separately.

    """
    def __init__(self, from_dict=None, from_list=None):
        super().__init__(from_dict=from_dict, from_list=from_list)

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


