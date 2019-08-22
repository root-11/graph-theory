import random

from graph import Graph


def random_xy_graph(nodes, x_max, y_max, edges=None, seed=42):
    """ Generates a graph with N nodes, M links, where all nodes have x,y in
    range [1,1] to [x_max, y_max]
    :param nodes: integer
    :param x_max: integer (800 pixels for example)
    :param y_max: integer (400 pixels for example)
    :param edges: integer or None, if None the graph will be fully connected.
    :param seed: seed for random number generator
    :return: Graph
    """

    def xy_distance(g, n1, n2):
        assert isinstance(g, Graph)
        x1, y1 = g.node(n1)
        x2, y2 = g.node(n2)
        return (abs(y2 - y1) + abs(x2 - x1)) ** (1 / 2)

    if x_max * y_max < nodes:
        raise ValueError("frame (x:{},y:{}) is too small for {} nodes".format(x_max,y_max,nodes))

    max_edges = sum(list(range(nodes))) * 2
    if edges is None:
        edges = max_edges
    if max_edges < edges:
        raise ValueError(
            "A fully connected graph with {} nodes, would at most have {} edges: {}".format(
                nodes, max_edges, edges
            ))

    random.seed(seed)
    g = Graph()
    xy_space = set()
    node_list = list(range(nodes))
    # random search mode
    node_count = 0
    while node_count < nodes:
        xy = (random.randint(1, x_max), random.randint(1, y_max))
        if xy not in xy_space:
            g.add_node(node_count, obj=xy)
            xy_space.add(xy)
            node_count += 1
            continue

        if len(xy_space) > (x_max * y_max) / 2:
            # use of random is inefficient.
            break

    # structured search mode.
    if len(g.nodes()) < nodes:
        x_range = list(range(1, x_max+1))
        random.shuffle(x_range)
        y_range = list(range(1, y_max+1))
        random.shuffle(y_range)
        quit = False
        for x in x_range:
            if quit:
                break
            for y in y_range:
                xy = (x, y)
                if xy in xy_space:
                    continue
                else:
                    g.add_node(node_count, obj=xy)
                    xy_space.add(xy)
                    node_count += 1

                if len(xy_space) == nodes:
                    quit = True
                    break

    n1s = node_list[:]
    n2s = set(node_list)
    edge_set = set()
    edge_count = 0
    while edge_count < edges:
        n1 = random.choice(n1s)
        existing_edges = set(g.nodes(from_node=n1))
        existing_edges.add(n1)
        candidates = list(n2s - existing_edges)
        if not candidates:
            n1s.remove(n1)
            continue
        n2 = random.choice(candidates)

        edge_set.add((n1, n2))
        d = xy_distance(g, n1, n2)
        g.add_edge(n1, n2, d)
        edge_count += 1

    return g