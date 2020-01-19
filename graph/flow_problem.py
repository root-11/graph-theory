from graph import BasicGraph
from graph.search import shortest_path


def maximum_flow(graph, start, end):
    """
    Returns the maximum flow graph
    :param graph: instance of Graph
    :param start: node
    :param end: node
    :return: flow, graph
    """
    assert isinstance(graph, BasicGraph)
    inflow = sum(d for s, e, d in graph.edges(from_node=start))
    outflow = sum(d for s, e, d in graph.edges(to_node=end))
    unassigned_flow = min(inflow, outflow)  # search in excess of this 'flow' is a waste of time.
    total_flow = 0
    # -----------------------------------------------------------------------
    # The algorithm
    # I reviewed a number of algorithms, such as Ford-fulkerson algorithm,
    # Edmonson-Karp and Dinic, but I didn't like them due to their naive usage
    # of BFS, which leads to a lot of node visits.
    #
    # I therefore choose to invert the capacities of the graph so that the
    # capacity any G[u][v] = c becomes 1/c in G_inverted.
    # This allows me to use the shortest path method to find the path with
    # most capacity in the first attempt, resulting in a significant reduction
    # of unassigned flow.
    #
    # By updating G_inverted, with the residual capacity, I can keep using the
    # shortest path, until the capacity is zero, whereby I remove the links
    # When the shortest path method returns 'No path' or when unassigned flow
    # is zero, I exit the algorithm.
    #
    # Even on small graphs, this method is very efficient, despite the overhead
    # of using shortest path. For very large graphs, this method outperforms
    # all other algorithms by orders of magnitude.
    # -----------------------------------------------------------------------

    edges = [(n1, n2, 1 / d) for n1, n2, d in graph.edges() if d > 0]
    inverted_graph = BasicGraph(from_list=edges)  # create G_inverted.
    capacity_graph = BasicGraph()  # Create structure to record capacity left.
    flow_graph = BasicGraph()  # Create structure to record flows.

    while unassigned_flow:
        # 1. find the best path
        d, path = shortest_path(inverted_graph, start, end)
        if d == float('inf'):  # then there is no path, and we must exit.
            return total_flow, flow_graph
        # else: use the path and lookup the actual flow from the capacity graph.

        path_flow = min([min(d, capacity_graph.edge(s, e, default=float('inf')))
                         for s, e, d in graph.edges(path=path)])

        # 2. update the unassigned flow.
        unassigned_flow -= path_flow
        total_flow += path_flow

        # 3. record the flows and update the inverted graph, so that it is
        #    ready for the next iteration.
        edges = graph.edges(path)
        for n1, n2, d in edges:

            # 3.a. recording:
            v = flow_graph.edge(n1, n2, default=None)
            if v is None:
                flow_graph.add_edge(n1, n2, path_flow)
                c = graph.edge(n1, n2) - path_flow
            else:
                flow_graph.add_edge(n1, n2, value=v + path_flow)
                c = graph.edge(n1, n2) - (v + path_flow)
            capacity_graph.add_edge(n1, n2, c)

            # 3.b. updating:
            # if there is capacity left: update with new 1/capacity
            # else: remove node, as we can't do 1/zero.
            if c > 0:
                inverted_graph.add_edge(n1, n2, 1 / c)
            else:
                inverted_graph.del_edge(n1, n2)
    return total_flow, flow_graph

