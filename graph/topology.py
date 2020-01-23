from graph import BasicGraph


def subgraph(graph, nodes):
    """ Creates a subgraph as a copy from the graph
    :param graph: class Graph
    :param nodes: list of nodes
    :return: new instance of Graph.
    """
    assert isinstance(graph, BasicGraph)
    assert isinstance(nodes, list)
    G = object.__new__(graph.__class__)
    G.__init__()
    for n1 in nodes:
        G.add_node(n1)
        for n2 in graph.nodes(from_node=n1):
            G.add_edge(n1, n2, graph.edge(n1, n2))
    return G


def is_subgraph(graph1, graph2):
    """
    Checks if graph1 is subgraph in graph2
    :param graph1: instance of Graph
    :param graph2: instance of Graph
    :return: boolean
    """
    assert isinstance(graph1, BasicGraph)
    assert isinstance(graph2, BasicGraph)
    if not set(graph1.nodes()).issubset(set(graph2.nodes())):
        return False
    if not set(graph1.edges()).issubset(set(graph2.edges())):
        return False
    return True


def is_partite(graph, n):
    """ Checks if graph is n-partite
    :param graph: class Graph
    :param n: int, number of partitions.
    :return: boolean and partitions as dict[colour] = set(nodes) or None.
    """
    assert isinstance(graph, BasicGraph)
    assert isinstance(n, int)
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


def same(path1, path2):
    """ Compares two paths to verify whether they're the same.
    :param path1: list of nodes.
    :param path2: list of nodes.
    :return: boolean.
    """
    start1 = path2.index(path1[0])
    checks = [
        path1[:len(path1) - start1] == path2[start1:],
        path1[len(path1) - start1:] == path2[:start1]
    ]
    if all(checks):
        return True
    return False


def has_path(graph, path):
    """ checks if path exists is graph
    :param graph: instance of Graph
    :param path: list of nodes
    :return: boolean
    """
    assert isinstance(graph, BasicGraph)
    assert isinstance(path, list)
    v1 = path[0]
    for v2 in path[1:]:
        if graph.edge(v1, v2) is None:
            return False
        else:
            v1 = v2
    return True


def has_cycles(graph):
    """ Checks if graph has a cycle
    :param graph: instance of class Graph.
    :return: bool
    """
    for n1, n2, d in graph.edges():
        if n1 == n2:
            return True
        if graph.depth_first_search(start=n2, end=n1):
            return True
    return False


def components(graph):
    """ Determines the components of the graph
    :param graph: instance of class Graph
    :return: list of sets of nodes. Each set is a component.
    """
    assert isinstance(graph, BasicGraph)
    nodes = set(graph.nodes())
    sets_of_components = []
    while nodes:
        new_component = set()
        sets_of_components.append(new_component)
        n = nodes.pop()  # select random node
        new_component.add(n)  # add it to the new component.

        new_nodes = set(graph.nodes(from_node=n))
        new_nodes.update(set(graph.nodes(to_node=n)))
        while new_nodes:
            n = new_nodes.pop()
            new_component.add(n)
            new_nodes.update(set(n for n in graph.nodes(from_node=n) if n not in new_component))
            new_nodes.update(set(n for n in graph.nodes(to_node=n) if n not in new_component))
        nodes = nodes - new_component
    return sets_of_components


def network_size(graph, n1, degrees_of_separation=None):
    """ Determines the nodes within the range given by
    a degree of separation
    :param graph: Graph
    :param n1: start node
    :param degrees_of_separation: integer
    :return: set of nodes within given range
    """
    assert isinstance(graph, BasicGraph)
    assert n1 in graph.nodes()
    if degrees_of_separation is not None:
        assert isinstance(degrees_of_separation, int)

    network = {n1}
    q = set(graph.nodes(from_node=n1))

    scan_depth = 1
    while True:
        if not q:  # then there's no network.
            break

        if degrees_of_separation is not None:
            if scan_depth > degrees_of_separation:
                break

        new_q = set()
        for peer in q:
            if peer in network:
                continue
            else:
                network.add(peer)
                new_peers = set(graph.nodes(from_node=peer)) - network
                new_q.update(new_peers)
        q = new_q
        scan_depth += 1
    return network


def phase_lines(graph):
    """ Determines the phase lines of a directed graph.
    :param graph: Graph
    :return: dictionary with node id : phase in cut.
    """
    phases = {n: 0 for n in graph.nodes()}
    q = graph.nodes(in_degree=0)
    q_set = set(q)
    seen = set()
    while q:
        n = q.pop(0)
        if n in seen:
            continue  # the graph is cyclic.
        seen.add(n)
        q_set.remove(n)
        level = phases[n]
        children = graph.nodes(from_node=n)
        for c in children:
            if phases[c] <= level:
                phases[c] = level + 1
            if c not in q_set:
                q.append(c)
                q_set.add(c)
    return phases


def sources(graph, n):
    """ Determines the set of sources of 'node' in a DAG
    :param graph: Graph
    :return: set of nodes
    """
    nodes = {n}
    q = [n]
    while q:
        new = q.pop(0)
        for src in graph.nodes(to_node=new):
            if src not in nodes:
                nodes.add(src)
            if src not in q:
                q.append(src)
    nodes.remove(n)
    return nodes




