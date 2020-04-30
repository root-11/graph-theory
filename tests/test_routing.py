from graph import Graph
from graph.routing import CVRPTW, Load, Vehicle
from itertools import permutations, chain


def make_road_network():
    edges = [(a, b, 1/24)  # 1 hour transit time.
             for a, b in permutations((0, 1, 2, 3, 4, 5), 2)]
    return Graph(from_list=edges)


def test_cvrptw():
    cvrptw = CVRPTW(graph=make_road_network())
    vehicle1 = Vehicle(start=0, capacity=1, endurance=8 / 24)
    vehicle2 = Vehicle(start=0, capacity=1, endurance=8 / 24)
    vehicles = [vehicle1, vehicle2]
    load1 = Load(start=0, end=1, size=0.3, open=6 / 24, close=8 / 24, handling_time=30/(60*60*24))
    load2 = Load(start=0, end=2, size=0.3, open=6 / 24, close=8 / 24, handling_time=30/(60*60*24))
    load3 = Load(start=0, end=3, size=0.3, open=6 / 24, close=8 / 24, handling_time=30/(60*60*24))
    load4 = Load(start=0, end=4, size=0.3, open=6 / 24, close=8 / 24, handling_time=30/(60*60*24))
    load5 = Load(start=0, end=5, size=0.3, open=6 / 24, close=8 / 24, handling_time=30/(60*60*24))
    loads = [load1, load2, load3, load4, load5]
    for i in chain(loads, vehicles):
        cvrptw.add(i)
    cvrptw.solve()
    print(cvrptw)


