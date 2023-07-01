from .base import BasicGraph, subgraph, is_subgraph, same_path, has_path, network_size, components
from .adjacency_matrix import adjacency_matrix
from .all_pairs_shortest_path import all_pairs_shortest_paths
from .all_paths import all_paths
from .all_simple_paths import all_simple_paths
from .bfs import breadth_first_search, breadth_first_walk
from .critical_path import critical_path_minimize_for_slack, critical_path
from .cycle import cycle, has_cycles
from .dag import phase_lines, sources
from .degree_of_separation import degree_of_separation
from .dfs import depth_first_search, depth_scan
from .distance_map import distance_map
from .max_flow import maximum_flow
from .max_flow_min_cut import maximum_flow_min_cut
from .min_cost_flow import minimum_cost_flow_using_successive_shortest_path
from .minmax import minmax
from .minsum import minsum
from .partite import is_partite
from .shortest_path import shortest_path, shortest_path_bidirectional, ShortestPathCache, distance_from_path
from .shortest_tree_all_pairs import shortest_tree_all_pairs
from .topological_sort import topological_sort
from .tsp import tsp_branch_and_bound, tsp_greedy, tsp_2023


__description__ = """
The graph-theory library is organised in the following way for clarity of structure:

1. BasicGraph (class) - with general methods for all subclasses.
2. All methods for class Graph in same order as on Graph.
3. Graph (class)
4. Graph3D (class) 
"""


class Graph(BasicGraph):
    """
    Graph is the base graph that all methods use.

    For methods, please see the documentation on the
    individual functions, by importing them separately.

    """

    def __init__(self, from_dict=None, from_list=None):
        super().__init__(from_dict=from_dict, from_list=from_list)
        self._cache = None

    def shortest_path(self, start, end, memoize=False, avoids=None):
        """
        :param start: start node
        :param end: end node
        :param memoize: boolean (stores paths in a cache for faster repeated lookup)
        :param avoids: optional. A frozen set of nodes that cannot be on the path.
        :return: distance, path as list
        """
        if not memoize:
            return shortest_path(graph=self, start=start, end=end, avoids=avoids)

        if self._cache is None:
            self._cache = ShortestPathCache(graph=self)
        return self._cache.shortest_path(start, end, avoids=avoids)

    def shortest_path_bidirectional(self, start, end):
        """
        :param start: start node
        :param end: end node
        :return: distance, path as list
        """
        return shortest_path_bidirectional(self, start, end)

    def breadth_first_search(self, start, end):
        """Determines the path with fewest nodes.
        :param start: start node
        :param end: end nodes
        :return: nodes, path as list
        """
        return breadth_first_search(graph=self, start=start, end=end)

    def breadth_first_walk(self, start, end=None, reversed_walk=False):
        """
        :param start: start node
        :param end: end node
        :param reversed_walk: if True, the BFS walk is backwards.
        :return: generator for breadth-first walk
        """
        return breadth_first_walk(graph=self, start=start, end=end, reversed_walk=reversed_walk)

    def distance_map(self, starts=None, ends=None, reverse=False):
        """Maps the shortest path distance from any start to any end.
        :param graph: instance of Graph
        :param starts: node or (set,list,tuple) of nodes
        :param ends: None (exhaustive map), node or (set,list,tuple) of nodes
        :param reverse: boolean, if True follows edges backwards.
        :return: dictionary with {node: distance from start}
        """
        return distance_map(self, starts, ends, reverse)

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
        return distance_from_path(graph=self, path=path)

    def maximum_flow(self, start, end):
        """Determines the maximum flow of the graph between
        start and end.
        :param start: node (source)
        :param end: node (sink)
        :return: flow, graph of flow.
        """
        return maximum_flow(self, start, end)

    def maximum_flow_min_cut(self, start, end):
        """
        Finds the edges in the maximum flow min cut.
        :param start: start
        :param end: end
        :return: list of edges
        """
        return maximum_flow_min_cut(self, start, end)

    def minimum_cost_flow(self, inventory, capacity=None):
        """
        :param self: Graph with `cost per unit` as edge
        :param inventory: dict {node: stock, ...}
            stock < 0 is demand
            stock > 0 is supply
        :param capacity: None or Graph with `capacity` as edge.
        :return: total costs, graph of flows in solution.
        """
        return minimum_cost_flow_using_successive_shortest_path(self, inventory, capacity)

    def solve_tsp(self, method="greedy"):
        """solves the traveling salesman problem for the graph
        (finds the shortest path through all nodes)

        :param method: str: 'greedy'

        options:
            'greedy' see tsp_greedy
            'bnb' see tsp_branch_and_bound

        :return: tour length (path+return to starting point),
                 path travelled.
        """
        methods = {"greedy": tsp_greedy, "bnb": tsp_branch_and_bound, "2023": tsp_2023}
        solver = methods.get(method, "tsp_2023")
        return solver(self)

    def subgraph_from_nodes(self, nodes):
        """
        constructs a copy of the graph containing only the
        listed nodes (and their links)
        :param nodes: list of nodes
        :return: class Graph
        """
        return subgraph(graph=self, nodes=nodes)

    def is_subgraph(self, other):
        """Checks if self is a subgraph in other.
        :param other: instance of Graph
        :return: boolean
        """
        return is_subgraph(self, other)

    def is_partite(self, n=2):
        """Checks if self is n-partite
        :param n: int the number of partitions.
        :return: tuple: boolean, partitions as dict
                        (or None if graph isn't n-partite)
        """
        return is_partite(self, n)

    def has_cycles(self):
        """Checks if the graph has a cycle
        :return: bool
        """
        return has_cycles(graph=self)

    def components(self):
        """Determines the number of components
        :return: list of sets of nodes. Each set is a component.
        """
        return components(graph=self)

    def network_size(self, n1, degrees_of_separation=None):
        """Determines the nodes within the range given by
        a degree of separation
        :param n1: start node
        :param degrees_of_separation: integer
        :return: set of nodes within given range
        """
        return network_size(self, n1, degrees_of_separation)

    def phase_lines(self):
        """Determines the phase lines (cuts) of the graph
        :returns: dictionary with phase: nodes in phase
        """
        return phase_lines(self)

    def sources(self, n):
        """Determines the DAG sources of node n"""
        return sources(graph=self, n=n)

    def topological_sort(self, key=None):
        """Returns a generator for the topological order"""
        return topological_sort(self, key=key)

    def critical_path(self):
        f"""{critical_path.__doc__}"""
        return critical_path(self)

    def critical_path_minimize_for_slack(self):
        f"""{critical_path_minimize_for_slack.__doc__}"""
        return critical_path_minimize_for_slack(self)

    @staticmethod
    def same_path(p1, p2):
        """compares two paths to determine if they're the same, despite
        being in different order.

        :param p1: list of nodes
        :param p2: list of nodes
        :return: boolean
        """
        return same_path(p1, p2)

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
        """Finds the mode(s) that have the smallest sum of distance to all other nodes.
        :return: list of nodes
        """
        return minsum(self)

    def minmax(self):
        """Finds the node(s) with shortest distance to all other nodes.
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

    def all_simple_paths(self, start, end):
        """
        finds all simple (non-looping) paths from start to end
        :param start: node
        :param end: node
        :return: list of paths
        """
        return all_simple_paths(self, start, end)

    def all_paths(self, start, end):
        """finds all paths from start to end by traversing each fork once only.
        :param start: node
        :param end: node
        :return: list of paths
        """
        return all_paths(graph=self, start=start, end=end)

    def degree_of_separation(self, n1, n2):
        """determines the degree of separation between 2 nodes
        :param n1: node
        :param n2: node
        :return: degree
        """
        return degree_of_separation(self, n1, n2)

    def loop(self, start, mid, end=None):
        """finds a looped path via a mid-point
        :param start: node
        :param mid: node, midpoint for loop.
        :param end: node
        :return: path as list
        """
        return cycle(self, start, mid, end)


class Graph3D(Graph):
    """a graph where all (x,y)-positions are unique."""

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
        """returns the distance between to xyz tuples coordinates
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
        """returns the node id of the `n` nearest neighbours."""
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

    def plot(self, nodes=True, edges=True, rotation="xyz", maintain_aspect_ratio=False):
        """plots nodes and links using matplotlib3
        :param nodes: bool: plots nodes
        :param edges: bool: plots edges
        :param rotation: str: set view point as one of [xyz,xzy,yxz,yzx,zxy,zyx]
        :param maintain_aspect_ratio: bool: rescales the chart to maintain aspect ratio.
        :return: None. Plots figure.
        """
        from graph.visuals import plot_3d  # noqa

        return plot_3d(self, nodes, edges, rotation, maintain_aspect_ratio)
