import random
import itertools

from graph import Graph


def xy_distance(n1, n2):
    """ calculates the xy_distance between to (x,y)-nodes"""
    x1, y1 = n1
    x2, y2 = n2
    dy, dx = (y2 - y1) * (y2 - y1), (x2 - x1) * (x2 - x1)
    return (dy + dx) ** (1 / 2)


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
    if x_max * y_max < nodes:
        raise ValueError("frame (x:{},y:{}) is too small for {} nodes".format(x_max,y_max,nodes))

    max_edges = nodes * nodes
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

    # Step 1: random search mode
    node_count = 0
    while node_count < nodes:
        xy = (random.randint(1, x_max), random.randint(1, y_max))
        if xy not in xy_space:
            g.add_node(xy)
            xy_space.add(xy)
            node_count += 1
            continue

        if len(xy_space) > (x_max * y_max) / 2:
            # use of random is inefficient --> proceed with step 2.
            break

    # Step 2: structured search mode.
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
                    g.add_node(xy)
                    xy_space.add(xy)
                    node_count += 1

                if len(xy_space) == nodes:
                    quit = True
                    break

    n1s = g.nodes()
    random.shuffle(n1s)
    n2s = n1s[:]
    random.shuffle(n2s)

    edge_count = 0
    for n1, n2 in itertools.product(*[n1s, n2s]):
        if edge_count == edges:
            break
        edge_count += 1

        d = xy_distance(n1, n2)
        g.add_edge(n1, n2, d)

    return g