from .base import BasicGraph


def is_partite(graph, n):
    """Checks if graph is n-partite
    :param graph: class Graph
    :param n: int, number of partitions.
    :return: boolean and partitions as dict[colour] = set(nodes) or None.
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")

    if not isinstance(n, int):
        raise TypeError(f"Expected n as integer > 0, not {type(n)}")
    colours_and_nodes = {i: set() for i in range(n)}
    nodes_and_colours = {}
    n1 = set(graph.nodes()).pop()
    q = [n1]
    visited = set()
    colour = 0
    while q:
        n1 = q.pop()
        visited.add(n1)

        if n1 in nodes_and_colours:
            colour = nodes_and_colours[n1]
        else:
            colours_and_nodes[colour].add(n1)
            nodes_and_colours[n1] = colour

        next_colour = (colour + 1) % n
        neighbours = graph.nodes(from_node=n1) + graph.nodes(to_node=n1)
        for n2 in neighbours:
            if n2 in nodes_and_colours:
                if nodes_and_colours[n2] == colour:
                    return False, None
                # else:  pass  # it already has a colour and there is no conflict.
            else:  # if n2 not in nodes_and_colours:
                colours_and_nodes[next_colour].add(n2)
                nodes_and_colours[n2] = next_colour
                continue
            if n2 not in visited:
                q.append(n2)

    return True, colours_and_nodes