try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # import required by matplotlib.
    visuals_enabled = True
except ImportError:
    visuals_enabled = False


def visualise(func):
    def wrapper(*args, **kwargs):
        if not visuals_enabled:
            raise ImportError("visualise is not available unless matplotlib is installed")
        return func(*args, **kwargs)
    return wrapper


@visualise
def plot_2d(graph, nodes=True, edges=True):
    """
    :param graph: instance of Graph with nodes as (x,y)
    :param nodes: bool: plots nodes
    :param edges: bool: plots edges
    :return: matlibplot.pyplot

    PRO-TIP: If your graph does not have nodes as (x,y) use random_xy_graph to
    create it with this recipe:

    Step 1: Get the imports.
    >>> from graph.random import random_xy_graph
    >>> from graph.hash import graph_hash

    Step 2: Determine the initialisation values.
    >>> node_count = len(graph.nodes())
    >>> seed = graph_hash(graph)
    >>> x_max = node_count * 8
    >>> y_max = node_count * 4
    >>> xygraph = random_xy_graph(node_count, x_max, y_max, edges=0, seed=seed)

    Step 3: create a mapping between the xy graph and the original graph.
    >>> mapping = {a:b for a,b in zip(xygraph.nodes(), graph.nodes())}

    Step 4: add the edges:
    >>> for edge in graph.edges():
    >>>     start, end, distance = edge
    >>>     xygraph.add_edge(mapping[start], mapping[end], distance)
    >>> plt = xygraph.plot_2d()
    >>> plt.show()

    """
    assert isinstance(nodes, bool)
    assert isinstance(edges, bool)

    for node in graph.nodes():
        if not isinstance(node, tuple):
            raise ValueError(f"expected graph.nodes() to be tuple(x,y), but found {node}")
        if not len(node) == 2:
            raise ValueError(f"expected tuples have 2 values, but found {node} (len={len(node)})")
        x, y = node
        if not isinstance(x, (float, int)):
            raise ValueError(f"expected node in graph.nodes() to have (x,y) as float or int, but got {type(x)}")
        if not isinstance(y, (float, int)):
            raise ValueError(f"expected node in graph.nodes() to have (x,y) as float or int, but got {type(y)}")

    plt.figure()
    if nodes:
        xs, ys = [a[0] for a in graph.nodes()], [a[1] for a in graph.nodes()]
        plt.plot(xs, ys)

    if edges:
        for edge in graph.edges():
            s, e, d = edge  # s: (x1,y1), e: (x2,y2), d: distance
            plt.plot([s[0], e[0]], [s[1], e[1]], 'bo-', clip_on=False)

    plt.axis('scaled')
    plt.axis('off')
    return plt


@visualise
def plot_3d(graph, nodes=True, edges=True, rotation='xyz', maintain_aspect_ratio=False):
    """ plots nodes and links using matplotlib3
    :param nodes: bool: plots nodes
    :param edges: bool: plots edges
    :param rotation: str: set view point as one of [xyz,xzy,yxz,yzx,zxy,zyx]
    :param maintain_aspect_ratio: bool: rescales the chart to maintain aspect ratio.
    :return: matlibplot.pyplot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if not len(rotation) == 3:
        raise ValueError(f"expected viewpoint as 'xyz' but got: {rotation}")
    for c in 'xyz':
        if c not in rotation:
            raise ValueError(f"rotation was missing {c}.")
    x, y, z = rotation

    # Data for a three-dimensional line
    if edges:
        for edge in graph.edges():
            n1, n2, v = edge
            xyz = dict()
            xyz[x] = [n1[0], n2[0]]
            xyz[y] = [n1[1], n2[1]]
            xyz[z] = [n1[2], n2[2]]
            ax.plot3D(xyz['x'], xyz['y'], xyz['z'], 'gray')

    # Data for three-dimensional scattered points
    if nodes:
        xyz = {x: [], y: [], z: []}
        ix = []
        for idx, node in enumerate(graph.nodes()):
            vx, vy, vz = node  # value of ...
            xyz[x].append(vx)
            xyz[y].append(vy)
            xyz[z].append(vz)
            ix.append(idx)
        ax.scatter3D(xyz['x'], xyz['y'], xyz['z'], c=ix, cmap='Greens')

    if (nodes or edges) and maintain_aspect_ratio:
        nodes = [n for n in graph.nodes()]
        xyz_dir = {'x': 0, 'y': 1, 'z': 2}

        xdim = xyz_dir[x]  # select the x dimension in the projection.
        # as the rotation will change the xdimension index.
        xs = [n[xdim] for n in nodes]  # use the xdim index to obtain the values.
        xmin, xmax = min(xs), max(xs)
        dx = (xmax + xmin) / 2  # determine the midpoint for the dimension.

        ydim = xyz_dir[y]
        ys = [n[ydim] for n in nodes]
        ymin, ymax = min(ys), max(ys)
        dy = (ymax + ymin) / 2

        zdim = xyz_dir[z]
        zs = [n[zdim] for n in nodes]
        zmin, zmax = min(zs), max(zs)
        dz = (zmax + zmin) / 2

        # calculate the radius for the aspect ratio.
        max_dim = max([xmax - xmin, ymax - ymin, zmax - zmin]) / 2

        xa, xb = dx - max_dim, dx + max_dim  # lower, uppper
        ax.set_xlim(xa, xb)  # apply the lower and upper to the axis.
        ya, yb = dy - max_dim, dy + max_dim
        ax.set_ylim(ya, yb)
        za, zb = dz - max_dim, dz + max_dim
        ax.set_zlim(za, zb)

    ax.set_xlabel(f'{x} Label')
    ax.set_ylabel(f'{y} Label')
    ax.set_zlabel(f'{z} Label')
    return plt
