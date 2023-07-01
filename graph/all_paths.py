from .base import BasicGraph


from collections import deque


def all_paths(graph, start, end):
    """finds all paths from start to end by traversing each fork once only.
    :param graph: instance of Graph
    :param start: node
    :param end: node
    :return: list of paths unique from start to end.
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected BasicGraph, Graph or Graph3D, not {type(graph)}")
    if start not in graph:
        raise ValueError("start not in graph.")
    if end not in graph:
        raise ValueError("end not in graph.")
    if start == end:
        raise ValueError("start is end")

    cache = {}
    if not graph.is_connected(start, end):
        return []
    paths = [(start,)]
    q = deque([start])
    skip_list = set()
    while q:
        n1 = q.popleft()
        if n1 == end:
            continue

        n2s = graph.nodes(from_node=n1)
        new_paths = [p for p in paths if p[-1] == n1]
        for n2 in n2s:
            if n2 in skip_list:
                continue
            n3s = graph.nodes(from_node=n2)

            con = cache.get((n2, n1))
            if con is None:
                con = graph.is_connected(n2, n1)
                cache[(n2, n1)] = con

            if len(n3s) > 1 and con:
                # it's a fork and it's a part of a loop!
                # is the sequence n2,n3 already in the path?
                for n3 in n3s:
                    for path in new_paths:
                        a = [n2, n3]
                        if any(all(path[i + j] == a[j] for j in range(len(a))) for i in range(len(path))):
                            skip_list.add(n3)

            for path in new_paths:
                if path in paths:
                    paths.remove(path)

                new_path = path + (n2,)
                if new_path not in paths:
                    paths.append(new_path)

            if n2 not in q:
                q.append(n2)

    paths = [list(p) for p in paths if p[-1] == end]
    return paths