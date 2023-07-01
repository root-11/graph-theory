from .base import BasicGraph
from .max_flow import maximum_flow


def maximum_flow_min_cut(graph, start, end):
    """
    Finds the edges in the maximum flow min cut.
    :param graph: Graph
    :param start: start
    :param end: end
    :return: list of edges
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    if start not in graph:
        raise ValueError(f"{start} not in graph")
    if end not in graph:
        raise ValueError(f"{end} not in graph")

    flow, mfg = maximum_flow(graph, start, end)
    if flow == 0:
        return []

    cls = type(graph)
    working_graph = cls(from_list=mfg.to_list())

    min_cut = []
    for n1 in mfg.breadth_first_walk(start, end):
        n2s = mfg.nodes(from_node=n1)
        for n2 in n2s:
            if graph.edge(n1, n2) - mfg.edge(n1, n2) == 0:
                working_graph.del_edge(n1, n2)
                min_cut.append((n1, n2))

    min_cut_nodes = set(working_graph.nodes(out_degree=0))
    min_cut_nodes.remove(end)
    min_cut = [(n1, n2) for (n1, n2) in min_cut if n1 in min_cut_nodes]
    return min_cut