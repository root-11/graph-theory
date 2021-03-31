# Graph-theory examples

This folder contains jupyter notebooks, with examples from
graph-theory.

If you're new to the field I would recommend to study (loosely)
in the order below:

***basic_graph_theory.ipynb***: An introduction to the basic 
terminology of graph-theory. Topics are:

	nodes
	edges
	indegree
	outdegree
	has_path
	distance path
	is a subgraph
	examples of existing graphs available for testing.

***generating and visualising graphs.***: An introduction
to making random xy graphs, grids and visualise them.

***comparing graphs***: An overview of methods for comparing 
graphs, such as:

    topological sort
	phase lines
	graph-hash
	flow_graph_hash
	merkle-tree


***[solving search problems](solving%20search%20problems.ipynb)***: An introduction to different
methods for findings paths, including:

	adjacency matrix
	BFS
	DFS
      DFScan
	bidi-BFS
	TSP
	[critical path method]
	find loops

***[calculating statistics about graphs](statistics%20on%20graphs.ipynb)*** provides an overview
of common analysis of graphs, such as:

	components
	has cycles
	network size
	is partite
	degree of separation


***[solving transport problems](solving%20search%20problems.ipynb)*** provides tools for a wide range 
of problems where discrete transport is essential.

	minmax
	minsum
	shortest_tree all pairs.
	scheduling problem
	traffic scheduling problem
		jam solver
	trans shipment problem (needs rewrite)


***[solving flow problems](solving%20flow%20problems.ipynb)*** provides tools for solving a 
wide range of problems where continuous flow are central.

	max flow
	max flow min cut
	min cost flow
	all_simple_paths
	all_paths
	

***solving assignment problems*** provides tools for solving
any kind of assignment problem.

	assignment problem
	wtap 


***representing systems as graphs*** provides a use case
for using `graph as finite state machine`.



