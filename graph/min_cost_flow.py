from .base import BasicGraph


from bisect import insort


def minimum_cost_flow_using_successive_shortest_path(costs, inventory, capacity=None):
    """
    Calculates the minimum cost flow solution using successive shortest path.
    :param costs: Graph with `cost per unit` as edge
    :param inventory: dict {node: stock, ...}
        stock < 0 is demand
        stock > 0 is supply
    :param capacity: None or Graph with `capacity` as edge.
        if capacity is None, capacity is assumed to be float('inf')
    :return: total costs, flow graph
    """
    if not isinstance(costs, BasicGraph):
        raise TypeError(f"expected costs as Graph, not {type(costs)}")
    Graph = type(costs)
    
    if not isinstance(inventory, dict):
        raise TypeError(f"expected inventory as dict, not {type(inventory)}")

    if not all(d >= 0 for s, e, d in costs.edges()):
        raise ValueError("The costs graph has negative edges. That won't work.")

    if not all(isinstance(v, (float, int)) for v in inventory.values()):
        raise TypeError("not all stock is numeric.")

    if capacity is None:
        capacity = Graph(from_list=[(s, e, float("inf")) for s, e, d in costs.edges()])
    else:
        if not isinstance(capacity, Graph):
            raise TypeError("Expected capacity as a Graph")
        if any(d < 0 for s, e, d in capacity.edges()):
            nn = [(s, e) for s, e, d in capacity.edges() if d < 0]
            raise ValueError(f"negative capacity on edges: {nn}")
        if {(s, e) for s, e, d in costs.edges()} != {(s, e) for s, e, d in capacity.edges()}:
            raise ValueError("cost and capacity have different links")

    # successive shortest path algorithm begins ...
    # ------------------------------------------------
    paths = costs.copy()  # initialise a copy of the cost graph so edges that
    # have exhausted capacities can be removed.
    flows = Graph()  # initialise F as copy with zero flow
    capacities = Graph()  # initialise C as a copy of capacity, so used capacity
    # can be removed.
    balance = [(v, k) for k, v in inventory.items() if v != 0]  # list with excess/demand, node id
    balance.sort()

    distances = paths.all_pairs_shortest_paths()

    while balance:  # while determine excess / imbalances:
        D, Dn = balance[0]  # pick node Dn where the demand D is greatest
        if D > 0:
            break  # only supplies left.
        balance = balance[1:]  # remove selection.

        supply_sites = [(distances[En][Dn], E, En) for E, En in balance if E > 0]
        if not supply_sites:
            break  # supply exhausted.
        supply_sites.sort()
        dist, E, En = supply_sites[0]  # pick nearest node En with excess E.
        balance.remove((E, En))  # maintain balance by removing the node.

        if E < 0:
            break  # no supplies left.
        if dist == float("inf"):
            raise Exception("bad logic: Case not checked for.")

        cost, path = paths.shortest_path(En, Dn)  # compute shortest path P from E to a node in demand D.

        # determine the capacity limit C on P:
        capacity_limit = min(capacities.edge(s, e, default=capacity.edge(s, e)) for s, e in zip(path[:-1], path[1:]))

        # determine L units to be transferred as min(demand @ D and the limit C)
        L = min(E, abs(D), capacity_limit)
        for s, e in zip(path[:-1], path[1:]):
            flows.add_edge(s, e, L + flows.edge(s, e, default=0))  # update F.
            new_capacity = capacities.edge(s, e, default=capacity.edge(s, e)) - L
            capacities.add_edge(s, e, new_capacity)  # update C

            if new_capacity == 0:  # remove the edge from potential solutions.
                paths.del_edge(s, e)
                distances = paths.all_pairs_shortest_paths()

        # maintain balance, in case there is excess or demand left.
        if E - L > 0:
            insort(balance, (E - L, En))
        if D + L < 0:
            insort(balance, (D + L, Dn))

    total_cost = sum(d * costs.edge(s, e) for s, e, d in flows.edges())
    return total_cost, flows