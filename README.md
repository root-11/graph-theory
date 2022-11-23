# graph-theory
![Build status](https://github.com/root-11/graph-theory/actions/workflows/python-test.yml/badge.svg)
[![codecov](https://codecov.io/gh/root-11/graph-theory/branch/master/graph/badge.svg?token=hWbKhIXskp)](https://codecov.io/gh/root-11/graph-theory)
[![Downloads](https://pepy.tech/badge/graph-theory)](https://pepy.tech/project/graph-theory)
[![Downloads](https://pepy.tech/badge/graph-theory/month)](https://pepy.tech/project/graph-theory/month)
[![PyPI version](https://badge.fury.io/py/graph-theory.svg)](https://badge.fury.io/py/graph-theory)


A simple graph library...<br>
*... A bit like networkx, just without the overhead...*<br> 
*... similar to graph-tool, without the Python 2.7 legacy...*<br>
*... with code that you can explain to your boss...*<br>

Detailed tutorial evolving in the [examples section](https://github.com/root-11/graph-theory/blob/master/examples/readme.md).
---------------------------

Latest features:

| date | description |
|---|---|
| 2022/10/04 | New tutorial: [Learn to solve traffic jams and sudoku's](https://github.com/root-11/graph-theory/blob/master/examples/graphs%20as%20finite%20state%20machines.ipynb) |
| 2022/03/09 | bugfixes to TrafficJamSolver only. |
| 2022/01/04 | new feature: Graph.distance_map, which allows the user to compute<br>the distance from a number of starts and ends as simulated annealing map. |
| 2022/01/04 | new generation of the traffic jam solver.|
| 2021/12/12 | shortest path now accepts keyword `avoids`, which allows the user<br>to declare nodes which cannot be a part of the path.<br>This feature has no impact on performance.|

---------------------------
Install:

    pip install graph-theory

Upgrade:

    pip install graph-theory --upgrade --no-cache

Testing:

    pytest tests --timesensitive  (for all tests)
    pytest tests (for logic tests only)

---------------------------
Import:

    import Graph
    g = Graph()  

    import Graph3d
    g3d = Graph3D()

---------------------------

Modules:

| module | description |
|:---|:---|
| `from graph import Graph, Graph3D` | Elementary methods (see basic methods below) for Graph and Graph3D.|
| `from graph import ...` | All methods available on Graph (see table below) |
| `from graph.assignment_problem import ...` | solvers for assignment problem, the Weapons-Target Assignment Problem, ... |
| `from graph.hash import ...` | graph hash functions: graph hash, merkle tree, flow graph hash | 
| `from graph.random import ...` | graph generators for random, 2D and 3D graphs. |
| `from graph.transshipment_problem import ...` | solvers for the transshipment problem |
| `from graph.traffic_scheduling_problem import ...` | solvers for the traffic jams (and slide puzzle) |
| `from graph.visuals import ...` | methods for creating matplotlib plots |
| `from graph.finite_state_machine import ...` | finite state machine |


All module functions are available from Graph and Graph3D (where applicable).

| Graph | Graph3D | methods                                          | returns                                                                                                                                                                                                               | example |
|:---:|:---:|:-------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| + | + | `a in g`                                         | assert if g contains node a                                                                                                                                                                                           | |
| + | + | `g.add_node(n, [obj])`                           | adds a node (with a pointer to object `obj` if given)                                                                                                                                                                 ||
| + | + | `g.copy()`                                       | returns a shallow copy of `g`                                                                                                                                                                                         ||
| + | + | `g.node(node1)`                                  | returns object attached to node 1                                                                                                                                                                                     ||
| + | + | `g.del_node(node1)`                              | deletes node1 and all it's edges                                                                                                                                                                                      ||
| + | + | `g.nodes()`                                      | returns a list of nodes                                                                                                                                                                                               ||
| + | + | `len(g.nodes())`                                 | returns the number of nodes                                                                                                                                                                                           ||
| + | + | `g.nodes(from_node=1)`                           | returns nodes with edges from node 1                                                                                                                                                                                  ||
| + | + | `g.nodes(to_node=2)`                             | returns nodes with edges to node 2                                                                                                                                                                                    ||
| + | + | `g.nodes(in_degree=2)`                           | returns nodes with 2 incoming edges                                                                                                                                                                                   ||
| + | + | `g.nodes(out_degree=2)`                          | returns nodes with 2 outgoing edges                                                                                                                                                                                   ||
| + | + | `g.add_edge(1,2,3)`                              | adds edge to g for vector `(1,2)` with value `3`                                                                                                                                                                      ||
| + | + | `g.edge(1,2)`                                    | returns value of edge between nodes 1 and 2                                                                                                                                                                           ||
| + | + | `g.edge(1,2,default=3)`                          | returns `default=3` if `edge(1,2)` doesn't exist. <br>similar to `d.get(key, 3)`                                                                                                                                      ||
| + | + | `g.del_edge(1,2)`                                | removes edge between nodes 1 and 2                                                                                                                                                                                    ||
| + | + | `g.edges()`                                      | returns a list of edges                                                                                                                                                                                               ||
| + | + | `len(g.edges())`                                 | returns the number of edges                                                                                                                                                                                           ||
| + | + | `g.edges(path=[path])`                           | returns a list of edges (along a path if given).                                                                                                                                                                      ||
| + | + | `same_path(p1,p2)`                               | compares two paths to determine if they contain same sequences <br>ex.: `[1,2,3] == [2,3,1]`                                                                                                                          ||
| + | + | `g.edges(from_node=1)`                           | returns edges outgoing from node 1                                                                                                                                                                                    ||
| + | + | `g.edges(to_node=2)`                             | returns edges incoming to node 2                                                                                                                                                                                      ||
| + | + | `g.from_dict(d)`                                 | updates the graph from a dictionary                                                                                                                                                                                   ||
| + | + | `g.to_dict()`                                    | returns the graph as a dictionary                                                                                                                                                                                     ||
| + | + | `g.from_list(L)`                                 | updates the graph from a list                                                                                                                                                                                         ||
| + | + | `g.to_list()`                                    | return the graph as a list of edges                                                                                                                                                                                   ||
| + | + | `g.shortest_path(start,end [, memoize, avoids])` | returns the distance and path for path with smallest edge sum <br> If `memoize=True`, sub results are cached for faster access if repeated calls.<br> If `avoids=set()`, then these nodes are not a part of the path. ||
| + | + | `g.shortest_path_bidirectional(start,end)`       | returns distance and path for the path with smallest edge sum using bidrectional search.                                                                                                                              ||
| + | + | `g.is_connected(start,end)`                      | determines if there is a path from start to end                                                                                                                                                                       ||
| + | + | `g.breadth_first_search(start,end)`              | returns the number of edges and path with fewest edges                                                                                                                                                                ||
| + | + | `g.breadth_first_walk(start,end)`                | returns a generator for a BFS walk                                                                                                                                                                                    ||
| + | + | `g.degree_of_separation(n1,n2)`                  | returns the distance between two nodes using BFS                                                                                                                                                                      ||
| + | + | `g.distance_map(starts,ends, reverse)`           | returns a dictionary with the distance from any start to any end (or reverse)                                                                                                                                         ||
| + | + | `g.network_size(n1, degree_of_separation)`       | returns the nodes within the range given by `degree_of_separation`                                                                                                                                                    ||
| + | + | `g.topological_sort(key)`                        | returns a generator that yields node in order from a non-cyclic graph.                                                                                                                                                ||
| + | + | `g.critical_path()`                              | returns the distance of the critical path and a list of Tasks.                                                                                                                                                        | [Example](examples/solving%20search%20problems.ipynb) |
| + | + | `g.critical_path_minimize_for_slack()`           | returns graph with artificial dependencies that minimises slack.                                                                                                                                                      | [Example](examples/solving%20search%20problems.ipynb)|
| + | + | `g.phase_lines()`                                | returns a dictionary with the phase_lines for a non-cyclic graph.                                                                                                                                                     ||
| + | + | `g.sources(n)`                                   | returns the source_tree of node `n`                                                                                                                                                                                   ||
| + | + | `g.depth_first_search(start,end)`                | returns path using DFS and backtracking                                                                                                                                                                               ||
| + | + | `g.depth_scan(start, criteria)`                  | returns set of nodes where criteria is True                                                                                                                                                                           ||
| + | + | `g.distance_from_path(path)`                     | returns the distance for path.                                                                                                                                                                                        ||
| + | + | `g.maximum_flow(source,sink)`                    | finds the maximum flow between a source and a sink                                                                                                                                                                    ||
| + | + | `g.maximum_flow_min_cut(source,sink)`            | finds the maximum flow minimum cut between a source and a sink                                                                                                                                                        ||
| + | + | `g.minimum_cost_flow(inventory, capacity)`       | finds the total cost and flows of the capacitated minimum cost flow.                                                                                                                                                  ||
| + | + | `g.solve_tsp()`                                  | solves the traveling salesman problem for the graph.<br>Available methods: 'greedy' (default) and 'bnb                                                                                                                ||
| + | + | `g.subgraph_from_nodes(nodes)`                   | returns the subgraph of `g` involving `nodes`                                                                                                                                                                         ||
| + | + | `g.is_subgraph(g2)`                              | determines if graph `g2` is a subgraph in g                                                                                                                                                                           ||
| + | + | `g.is_partite(n)`                                | determines if graph is n-partite                                                                                                                                                                                      ||
| + | + | `g.has_cycles()`                                 | determines if there are any cycles in the graph                                                                                                                                                                       ||
| + | + | `g.components()`                                 | returns set of nodes in each component in `g`                                                                                                                                                                         ||
| + | + | `g.same_path(p1,p2)`                             | compares two paths, returns True if they're the same                                                                                                                                                                  ||
| + | + | `g.adjacency_matrix()`                           | returns the adjacency matrix for the graph                                                                                                                                                                            ||
| + | + | `g.all_pairs_shortest_paths()`                   | finds the shortest path between all nodes                                                                                                                                                                             ||
| + | + | `g.minsum()`                                     | finds the node(s) with shortest total distance to all other nodes                                                                                                                                                     ||
| + | + | `g.minmax()`                                     | finds the node(s) with shortest maximum distance to all other nodes                                                                                                                                                   ||
| + | + | `g.shortest_tree_all_pairs()`                    | finds the shortest tree for all pairs                                                                                                                                                                                 ||
| + | + | `g.has_path(p)`                                  | asserts whether a path `p` exists in g                                                                                                                                                                                ||
| + | + | `g.all_simple_paths(start,end)`                  | finds all simple paths between 2 nodes                                                                                                                                                                                ||
| + | + | `g.all_paths(start,end)`                         | finds all combinations of paths between 2 nodes                                                                                                                                                                       ||
| - | + | `g3d.distance(n1,n2)`                            | returns the spatial distance between `n1` and `n2`                                                                                                                                                                    ||
| - | + | `g3d.n_nearest_neighbour(n1, [n])`               | returns the `n` nearest neighbours to node `n1`                                                                                                                                                                       ||
| - | + | `g3d.plot()`                                     | returns matplotlib plot of the graph.                                                                                                                                                                                 ||


## FAQ

| want to... | doesn't work... | do instead... | ...but why? |
|:---|:---|:---|:---|
| have multiple edges between two nodes | `Graph(from_list=[(1,2,3), (1,2,4)]` | Add dummy nodes<br>`[(1,a,3), (a,2,0),`<br>` (1,b,4),(b,2,0)]` | Explicit is better than implicit. |
| multiple values on an edge | `g.add_edge(1,2,{'a':3, 'b':4})` | Have two graphs<br>`g_a.add_edge(1,2,3)`<br>`g_b.add_edge(1,2,4)` | Most graph algorithms don't work with multiple values |
|do repeated calls to shortest path|`g.shortest_path(a,b)` is slow|Use `g.shortest_path(a,b,memoize=True)` instead|memoize uses bidirectional search and caches sub-results along the shortest path for future retrievals|

## Credits:

- Arturo Soucase for packaging and testing. 
- Peter Norvig for inspiration on TSP from [pytudes](https://github.com/norvig/pytudes/blob/master/ipynb/TSP.ipynb).
- Harry Darby for the mountain river map.
- Kyle Downey for depth_scan algorithm.
- Ross Blandford for munich firebrigade centre -, traffic jam - and slide puzzle - test cases.
- Avi Kelman for type-tolerant search, and a number of micro optimizations.
- Joshua Crestone for all simple paths test.
- CodeMartyLikeYou for detecting a bug in `@memoize` 
- Tom Carroll for detecting the bug in del_edge and inspiration for topological sort.

