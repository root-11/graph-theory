# graph

A simple graph library...

*...A bit like networkx, just without the overhead.*


---------------------------

    import Graph
    g = Graph()
    
That's it.

---------------------------

Available methods:

| methods | description |
|:---|:---|
| `a in g` | assert if g contains node a |
| `len(g)` | returns the number of nodes |
| `g.add_node(n, [obj])` | adds a node (with a pointer to object `obj` if given) |
| `g.node(node1)` | returns object attached to node 1. |
| `g.del_node(node1)` | deletes node1 and all it's links. |
| `g.nodes()` | returns a list of nodes |
| `g.nodes(from_node=1)` | returns nodes with edges from node 1 |
| `g.nodes(to_node=2)` | returns nodes with edges to node 2 |
| `g.nodes(in_degree=2)` | returns nodes with 2 incoming edges |
| `g.nodes(out_degree=2)` | returns nodes with 2 outgoing edges |
| `g.add_edge(1,2,3)` | adds edge to g for vector `(1,2)` with value `3` |
| `g[1][2]` | returns the distance/value/weight of vector `G[1][2]` in g |
| `g.edge(1,2)` | (alias) for `g[1][2]` above |
| `g.edge(1,2,default=3)` | returns `default=3` if `edge(1,2)` doesn't exist. <br>Similar to `d.get(key, 3)`|
| `g.del_edge(1,2)` | removes edge `G[1][2]` |
| `g.edges(path=[path])` | returns a list of edges (along a path if given). |
| `g.edges(node=1)` | returns edges outgoing from node 1 | 
| `g.from_dict(d)` | updates the graph from a dictionary of nodes |
| `g.to_dict()` | dumps the graph as a dictionary |
| `g.from_list(L)` | updates the graph from a list of edges `L=[(n1,n2,d), ...]` |
| `g.to_list()` | dumps the graph as a list of edges |
| `g.shortest_path(start,end)` | finds the path with smallest edge sum |
| `g.breadth_first_search(start,end)` | finds the with least number of hops |
| `g.depth_first_search(start,end)` | finds a path between 2 nodes (start, end) using DFS and backtracking. |
| `g.distance(path)` | finds the distance following a given path. |
| `g.maximum_flow(source,sink)` | finds the maximum flow between a source and a sink|
| `g.solve_tsp()` | solves the traveling salesman problem for the graph|
| `g.is_subgraph(g2)` | determines if graph `g2` is a subgraph in g.|
| `g.is_partite(n)` | determines if graph is n-partite |
| `g.adjacency_matrix()` | constructs the adjacency matrix for the graph.|
| `g.all_pairs_shortest_paths()` | finds the shortest path between all nodes. |
| `g.shortest_tree_all_pairs()` | finds the shortest tree for all pairs.|
| `g.has_path(p)` | asserts whether a path `p` exists in g.|
| `g.same_path(p1,p2)` | compares two paths, returns True if they're the same.|
| `g.all_paths(start,end)` | finds all combinations of paths between 2 nodes.|

| want to | doesn't work | do instead | but why? |
|:---|:---|:---|:---|
| add edge | `g[1][2] = 3` | `g.add_edge(1,2,3)` | `g[1][2]` retrieves the link. It's not for setting values. |
| update edge | `g[1][2] = 4` | `g.add_edge(1,2,4)` | See add edge |
| multiple edges | `Graph(from_list=[(1,2,3), (1,2,4)]` | Add dummy nodes<br>`[(1,a,3), (a,2,0),`<br>` (1,b,4),(b,2,0)]` | Explicit is better than implicit. |
| multiple values on edge | `g.add_edge(1,2,{'a':3, 'b':4})` | Have two graphs<br>`g_a.add_edge(1,2,3)`<br>`g_b.add_edge(1,2,4)` | Most graph algorithms don't work with multiple values |  
| add node | `g[1]` | `g.add_node(1)` | `g[1]` retrieves the node. It's not for assignment. | 
| add attribute to node | `g[1] = {'a':1, 'b':2}` | `g.add_node(1, obj={'a':1, 'b':2})` | see add_node |
| update node | `g[1]['c'] = 3` | `n = g.node(1)`<br>`n['c'] = 3` | Explicit is better than implicit. |
 

