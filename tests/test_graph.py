from graph import Graph


def graph01():
    """
    :return: Graph.
    """
    d = {1: {2: 10, 3: 5},
         2: {4: 1, 3: 2},
         3: {2: 3, 4: 9, 5: 2},
         4: {5: 4},
         5: {1: 7, 4: 6}}
    return Graph(from_dict=d)


def graph02():
    """
    1 -> 2 -> 3
    |    |    |
    v    v    v
    4 -> 5 -> 6
    |    |    |
    v    v    v
    7 -> 8 -> 9

    :return: :return:
    """
    d = {1: {2: 1, 4: 1},
         2: {3: 1, 5: 1},
         3: {6: 1},
         4: {5: 1, 7: 1},
         5: {6: 1, 8: 1},
         6: {9: 1},
         7: {8: 1},
         8: {9: 1}
         }
    return Graph(from_dict=d)


def graph03():
    d = {1: {2: 1, 3: 9, 4: 4, 5: 13, 6: 20},
         2: {1: 7, 3: 7, 4: 2, 5: 11, 6: 18},
         3: {8: 20, 4: 4, 5: 4, 6: 16, 7: 16},
         4: {8: 15, 3: 4, 5: 9, 6: 11, 7: 21},
         5: {8: 11, 6: 2, 7: 17},
         6: {8: 9, 7: 5},
         7: {8: 3},
         8: {7: 5}}
    return Graph(from_dict=d)


def graph04():
    d = {1: {2: 1, 3: 9, 4: 4, 5: 11, 6: 17},
         2: {1: 7, 3: 7, 4: 2, 5: 9, 6: 15},
         3: {8: 17, 4: 4, 5: 4, 6: 14, 7: 13},
         4: {8: 12, 3: 4, 5: 9, 6: 9, 7: 18},
         5: {8: 9, 6: 2, 7: 15},
         6: {8: 9, 7: 5},
         7: {8: 3},
         8: {7: 5}}
    return Graph(from_dict=d)


def graph05():
    """
    0 ---- 1 ---- 5
     +      +---- 6 ---- 7
      +            +     |
       +            +---- 8
        +
         +- 2 ---- 3 ---- 9
             +      +     |
              4      +---10
    """
    links = [
        (0, 1, 1),
        (0, 2, 1),
        (1, 5, 1),
        (1, 6, 1),
        (2, 3, 1),
        (2, 4, 1),
        (3, 9, 1),
        (3, 10, 1),
        (9, 10, 1),
        (6, 7, 1),
        (6, 8, 1),
        (7, 8, 1),
        (0, 1, 1),
        (0, 1, 1),
        (0, 1, 1),
    ]
    return Graph(from_list=links)


def graph_cycle_5():
    """ cycle of 5 nodes """
    links = [
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (4, 5, 1),
        (5, 1, 1),
    ]
    links.extend([(n2, n1, d) for n1, n2, d in links])
    return Graph(from_list=links)


def graph_cycle_6():
    """
    cycle of 6 nodes
    """
    links = [
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (4, 5, 1),
        (5, 6, 1),
        (6, 1, 1),
    ]
    links.extend([(n2, n1, d) for n1, n2, d in links])
    return Graph(from_list=links)


def fully_connected_4():
    """
    fully connected graph with 4 nodes.
    """
    L = [
        (0, 3, 1), (0, 2, 1), (0, 1, 1), (3, 2, 1), (3, 1, 1), (3, 0, 1),
        (2, 3, 1), (2, 1, 1), (2, 0, 1), (1, 0, 1), (1, 2, 1), (1, 3, 1),
        (0,), (1,), (2,), (3,)
    ]
    g = Graph(from_list=L)
    return g



