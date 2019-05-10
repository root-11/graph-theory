# graph

A simple graph library...<br>
*... A bit like networkx, just without the overhead...*<br> 
*... similar to graph-tool, without the Python 2.7 legacy...*<br>


---------------------------

    import Graph
    g = Graph()
    
That's it.

---------------------------

Available methods:

| methods | description |
|:---|:---|
| `a in g` | assert if g contains node a |
| `g.add_node(n, [obj])` | adds a node (with a pointer to object `obj` if given) |
| `g.node(node1)` | returns object attached to node 1. |
| `g.del_node(node1)` | deletes node1 and all it's edges. |
| `g.nodes()` | returns a list of nodes |
| `len(g.nodes())` | returns the number of nodes |
| `g.nodes(from_node=1)` | returns nodes with edges from node 1 |
| `g.nodes(to_node=2)` | returns nodes with edges to node 2 |
| `g.nodes(in_degree=2)` | returns nodes with 2 incoming edges |
| `g.nodes(out_degree=2)` | returns nodes with 2 outgoing edges |
| `g.add_edge(1,2,3)` | adds edge to g for vector `(1,2)` with value `3` |
| `g.edge(1,2)` | returns value of edge between nodes 1 and 2 |
| `g.edge(1,2,default=3)` | returns `default=3` if `edge(1,2)` doesn't exist. <br>similar to `d.get(key, 3)`|
| `g.del_edge(1,2)` | removes edge between nodes 1 and 2 |
| `g.edges(path=[path])` | returns a list of edges (along a path if given). |
| `g.edges(from_node=1)` | returns edges outgoing from node 1 |
| `g.edges(to_node=2)` | returns edges incoming to node 2 |
| `len(g.edges())` | returns the number of edges |
| `g.from_dict(d)` | updates the graph from a dictionary |
| `g.to_dict()` | dumps the graph as a dictionary |
| `g.from_list(L)` | updates the graph from a list |
| `g.to_list()` | dumps the graph as a list of edges |
| `g.shortest_path(start,end)` | finds the path with smallest edge sum |
| `g.breadth_first_search(start,end)` | finds the with least number of hops |
| `g.depth_first_search(start,end)` | finds a path between 2 nodes (start, end) using DFS and backtracking. |
| `g.distance_from_path(path)` | finds the distance following a given path. |
| `g.maximum_flow(source,sink)` | finds the maximum flow between a source and a sink|
| `g.solve_tsp()` | solves the traveling salesman problem for the graph|
| `g.is_subgraph(g2)` | determines if graph `g2` is a subgraph in g.|
| `g.is_partite(n)` | determines if graph is n-partite |
| `g.has_cycles()` | determines if there are cycles in the graph |
| `g.same_path(p1,p2)` | compares two paths, returns True if they're the same.|
| `g.adjacency_matrix()` | constructs the adjacency matrix for the graph.|
| `g.all_pairs_shortest_paths()` | finds the shortest path between all nodes. |
| `g.shortest_tree_all_pairs()` | finds the shortest tree for all pairs.|
| `g.has_path(p)` | asserts whether a path `p` exists in g.|
| `g.all_paths(start,end)` | finds all combinations of paths between 2 nodes.|

## FAQ

| want to | doesn't work | do instead | but why? |
|:---|:---|:---|:---|
| have multiple edges between two nodes | `Graph(from_list=[(1,2,3), (1,2,4)]` | Add dummy nodes<br>`[(1,a,3), (a,2,0),`<br>` (1,b,4),(b,2,0)]` | Explicit is better than implicit. |
| multiple values on an edge | `g.add_edge(1,2,{'a':3, 'b':4})` | Have two graphs<br>`g_a.add_edge(1,2,3)`<br>`g_b.add_edge(1,2,4)` | Most graph algorithms don't work with multiple values |   

## Examples

Examples contains a number of tutorial/solutions to common operations research
and computer science problems, which are made simple when treated as a graph.

| module | function | description |
|:---|:---|:---|
| assignment_problem.py | assignment_problem |  solves the assignment problem |
| hashgraph.py | merkle_tree | datablocks |
| hashgraph.py | graph_hash | computes the sha256 of a graphs nodes and edges |
| hashgraph.py | flow_graph_hash | computes the sha256 of a graph with multiple sources and sinks |
| knapsack_problem.py | knapsack problem | solves the knapsack problem |
| wtap.py | weapons-target assignment problem | solves the WTAP problem. | 

