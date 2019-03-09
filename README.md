# graph

A simple graph library...

...A bit like networkx, just without the overhead.


---------------------------

    import Graph
    g = Graph()
    
That's it.

Available methods:

| methods | description |
|:---|:---|
| `a in g` | assert if g contains node a |
| `len(g)` | returns the number of nodes |
| `g[1][2]` | returns the distance/value/weight of vector `G[1][2]` in g |
| `del g[1][2]` | removes edge `G[1][2]` |
| `g.nodes()` | returns a list of nodes |
| `g.edges([path])` | returns a list of edges (along a path if given). |
| `g.add_node` | adds a node |
|`g.add_edge` | adds edge to g |
|`g.from_dict` | updates the graph from a dictionary of nodes |
|`g.to_dict` | dumps the graph as a dictionary |
|`g.from_list` | updates the graph from a list of edges (n1,n2,d) |
|`g.to_list` | dumps the graph as a list of edges |
|`g.shortest_path` | finds shortest path between 2 nodes (start,end)|
|`g.breadth_first_search` | finds the least number of hops between 2 nodes (start,end) |
|`g.distance` | finds the distance following a given path. |
|`g.maximum_flow` | finds the maximum flow between a source and a sink|
|`g.solve_tsp` | solves the traveling salesman problem for the graph|
|`g.is_subgraph(g2)` | determines if graph `g2` is a subgraph in g.|
|`g.same_path` | compares two paths, returns True if they're the same.|
|`g.adjacency_matrix` | constructs the adjacency matrix for the graph.|
|`g.all_pairs_shortest_paths` | finds the shortest path between all nodes. |
|`g.shortest_tree_all_pairs` | finds the shortest tree for all pairs.|
|`g.has_path` | asserts whether a path exists in g.|
|`g.all_paths` | finds all combinations of paths between 2 nodes (start, end).|


