from graph import Graph


def test_maximum_flow():
    """ [2] ----- [5]
       /    +   /  | +
    [1]      [4]   |  [7]
       +    /   +  | /
        [3] ----- [6]
    """
    links = [
        (1, 2, 18),
        (1, 3, 10),
        (2, 4, 7),
        (2, 5, 6),
        (3, 4, 2),
        (3, 6, 8),
        (4, 5, 10),
        (4, 6, 10),
        (5, 6, 16),
        (5, 7, 9),
        (6, 7, 18)
    ]
    g = Graph(from_list=links)

    flow, g2 = g.maximum_flow(1, 7)
    assert flow == 23, flow


def test_maximum_flow01():
    links = [
        (1, 2, 1)
    ]
    g = Graph(from_list=links)
    flow, g2 = g.maximum_flow(start=1, end=2)
    assert flow == 1, flow


def test_maximum_flow02():
    links = [
        (1, 2, 10),
        (2, 3, 1),  # bottleneck.
        (3, 4, 10)
    ]
    g = Graph(from_list=links)
    flow, g2 = g.maximum_flow(start=1, end=4)
    assert flow == 1, flow


def test_maximum_flow03():
    links = [
        (1, 2, 10),
        (1, 3, 10),
        (2, 4, 1),  # bottleneck 1
        (3, 5, 1),  # bottleneck 2
        (4, 6, 10),
        (5, 6, 10)
    ]
    g = Graph(from_list=links)
    flow, g2 = g.maximum_flow(start=1, end=6)
    assert flow == 2, flow


def test_maximum_flow04():
    links = [
        (1, 2, 10),
        (1, 3, 10),
        (2, 4, 1),  # bottleneck 1
        (2, 5, 1),  # bottleneck 2
        (3, 5, 1),  # bottleneck 3
        (3, 4, 1),  # bottleneck 4
        (4, 6, 10),
        (5, 6, 10)
    ]
    g = Graph(from_list=links)
    flow, g2 = g.maximum_flow(start=1, end=6)
    assert flow == 4, flow


def test_maximum_flow05():
    links = [
        (1, 2, 10),
        (1, 3, 1),
        (2, 3, 1)
    ]
    g = Graph(from_list=links)
    flow, g2 = g.maximum_flow(start=1, end=3)
    assert flow == 2, flow