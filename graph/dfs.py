from .base import BasicGraph


from collections import deque


def depth_first_search(graph, start, end):
    """
    Determines path from start to end using
    'depth first search' with backtracking.
    :param graph: class Graph
    :param start: start node
    :param end: end node
    :return: path as list of nodes.
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    if start not in graph:
        raise ValueError(f"{start} not in graph")
    if end not in graph:
        raise ValueError(f"{end} not in graph")

    q = deque([start])  # q = [start]
    # using deque as popleft and appendleft is faster than using lists. For details see
    # https://stackoverflow.com/questions/23487307/python-deque-vs-list-performance-comparison
    path = []
    visited = set()
    while q:
        n1 = q.popleft()  # n1 = q.pop()
        visited.add(n1)
        path.append(n1)
        if n1 == end:
            return path  # <-- exit if end is found.
        for n2 in graph.nodes(from_node=n1):
            if n2 in visited:
                continue
            q.appendleft(n2)  # q.append(n2)
            break
        else:
            path.remove(n1)
            while not q and path:
                for n2 in graph.nodes(from_node=path[-1]):
                    if n2 in visited:
                        continue
                    q.appendleft(n2)  # q.append(n2)
                    break
                else:
                    path = path[:-1]
    return None  # <-- exit if not path was found.


def depth_scan(graph, start, criteria):
    """traverses the descendants of node `start` using callable `criteria` to determine
    whether to terminate search along each branch in `graph`.
    :param graph: class Graph
    :param start: start node
    :param criteria: function to terminate scan along a branch must return bool
    :return: set of nodes
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    if start not in graph:
        raise ValueError(f"{start} not in graph")
    if not callable(criteria):
        raise TypeError(f"Expected {criteria} to be callable")
    if not criteria(start):
        return set()

    q = [start]
    path = []
    visited = set()
    while q:
        n1 = q.pop()
        visited.add(n1)
        path.append(n1)
        for n2 in graph.nodes(from_node=n1):
            if n2 in visited:
                continue
            if not criteria(n2):
                visited.add(n2)
                continue
            q.append(n2)
            break
        else:
            path.remove(n1)
            while not q and path:
                for n2 in graph.nodes(from_node=path[-1]):
                    if n2 in visited:
                        continue
                    if not criteria(n2):
                        visited.add(n2)
                        continue
                    q.append(n2)
                    break
                else:
                    path = path[:-1]
    return visited