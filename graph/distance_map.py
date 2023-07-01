from .base import BasicGraph


from collections import deque
from collections.abc import Iterable


def distance_map(graph, starts=None, ends=None, reverse=False):
    """Maps the shortest path distance from any start to any end.
    :param graph: instance of Graph
    :param starts: None, node or (set,list,tuple) of nodes
    :param ends: None (exhaustive map), node or (set,list,tuple) of nodes
                 that terminate the search when all are found
    :param reverse: bool: walks the map from the ends towards the starts using reversed edges.
    :return: dictionary with {node: distance from start}
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")

    if not isinstance(reverse, bool):
        raise TypeError("keyword reverse was not boolean")

    if all((starts is None, ends is None)):
        raise ValueError("starts and ends cannot both be None")
    if all((starts is None, reverse is False)):
        raise ValueError("walking forward from end doesn't make sense.")
    if all((ends is None, reverse is True)):
        raise ValueError("walking from reverse from start doesn't make sense.")

    if isinstance(starts, Iterable):
        starts = set(starts)
    else:
        starts = {starts}
    if any(start not in graph for start in starts if start is not None):
        missing = [start not in graph for start in starts]
        raise ValueError(f"starts: ({missing}) not in graph")

    if isinstance(ends, Iterable):
        ends = set(ends)
    else:
        ends = {ends}
    if any(end not in graph for end in ends if end is not None):
        missing = [end not in graph for end in ends if end is not None]
        raise ValueError(f"{missing} not in graph. Use `end=None` if you want exhaustive search.")

    if not reverse:
        ends_found = set()
        visited = {start: 0 for start in starts}
        q = deque(starts)
        while q:
            if ends_found == ends:
                break
            n1 = q.popleft()
            if n1 in ends:
                ends_found.add(n1)
            d1 = visited[n1]
            for _, n2, d in graph.edges(from_node=n1):
                if n2 not in visited:
                    q.append(n2)
                visited[n2] = min(d1 + d, visited.get(n2, float("inf")))

    else:
        starts_found = set()
        visited = {end: 0 for end in ends}
        q = deque(ends)
        while q:
            if starts_found == starts:
                break
            n2 = q.popleft()
            if n2 in starts:
                starts_found.add(n2)
            d2 = visited[n2]
            for n1, _, d in graph.edges(to_node=n2):
                if n1 not in visited:
                    q.append(n1)
                visited[n1] = min(d + d2, visited.get(n1, float("inf")))

    return visited