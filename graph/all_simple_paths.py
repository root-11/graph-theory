from .base import BasicGraph


from collections import deque


def all_simple_paths(graph, start, end):
    """
    finds all simple (non-looping) paths from start to end
    :param start: node
    :param end: node
    :return: list of paths
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    if start not in graph:
        raise ValueError("start not in graph.")
    if end not in graph:
        raise ValueError("end not in graph.")
    if start == end:
        raise ValueError("start is end")

    if not graph.is_connected(start, end):
        return []

    paths = []
    q = deque([(start,)])
    while q:
        path = q.popleft()
        for s, e, d in graph.edges(from_node=path[0]):
            if e in path:
                continue
            new_path = (e,) + path
            if e == end:
                paths.append(new_path)
            else:
                q.append(new_path)
    return [list(reversed(p)) for p in paths]