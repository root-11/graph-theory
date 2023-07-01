from .base import BasicGraph


from collections import deque


def breadth_first_search(graph, start, end):
    """Determines the path from start to end with fewest nodes.
    :param graph: class Graph
    :param start: start node
    :param end: end node
    :return: path
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected BasicGraph, Graph or Graph3D, not {type(graph)}")
    if start not in graph:
        raise ValueError(f"{start} not in graph")
    if end not in graph:
        raise ValueError(f"{end} not in graph")

    visited = {start: None}
    q = deque([start])
    while q:
        node = q.popleft()
        if node == end:
            path = deque()
            while node is not None:
                path.appendleft(node)
                node = visited[node]
            return list(path)
        for next_node in graph.nodes(from_node=node):
            if next_node not in visited:
                visited[next_node] = node
                q.append(next_node)
    return []


def breadth_first_walk(graph, start, end=None, reversed_walk=False):
    """
    :param graph: Graph
    :param start: start node
    :param end: end node.
    :param reversed_walk: if True, the BFS traverse the graph backwards.
    :return: generator for walk.
    To walk all nodes use: `[n for n in g.breadth_first_walk(start)]`
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected BasicGraph, Graph or Graph3D, not {type(graph)}")
    if start not in graph:
        raise ValueError(f"{start} not in graph")
    if end is not None and end not in graph:
        raise ValueError(f"{end} not in graph. Use `end=None` if you want exhaustive search.")
    if not isinstance(reversed_walk, bool):
        raise TypeError(f"reversed_walk should be boolean, not {type(reversed_walk)}: {reversed_walk}")

    visited = {start: None}
    q = deque([start])
    while q:
        node = q.popleft()
        yield node
        if node == end:
            break
        L = graph.nodes(from_node=node) if not reversed_walk else graph.nodes(to_node=node)
        for next_node in L:
            if next_node not in visited:
                visited[next_node] = node
                q.append(next_node)