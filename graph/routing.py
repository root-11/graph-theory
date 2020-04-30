from graph import Graph
from itertools import count


class CVRPTW(object):
    """ solves the Capacitated Vehicle Routing Problem with Time Windows. """

    def __init__(self, graph):
        assert isinstance(graph, Graph)
        self.graph = graph
        self.am = self.graph.adjacency_matrix()
        self.vehicles = []
        self.loads = []

    def add(self, item):
        if isinstance(item, Load):
            self._new_load(item)
        elif isinstance(item, Vehicle):
            self._new_vehicle(item)
        else:
            raise NotImplementedError(f"no method for {type(item)}")

    def distance(self, a, b):
        return self.am[a][b]

    def _new_load(self, load):
        assert isinstance(load, Load)
        if load.from_loc not in self.graph.nodes():
            raise ValueError("start not in map")
        if load.to_loc not in self.graph.nodes():
            raise ValueError("end not in map")
        self.loads.append(load)

    def _new_vehicle(self, vehicle):
        assert isinstance(vehicle, Vehicle)
        if vehicle.start not in self.graph.nodes():
            raise ValueError("start not in map")
        self.vehicles.append(vehicle)
        vehicle.pd = self

    def solve(self):
        """ top level method for calling the solver"""
        # 1. create initial solution
        for vehicle in self.vehicles:
            vehicle.evaluate()
        # 2. swop until no further improvement
        pass

    def __str__(self):
        s = []
        sa = s.append
        for vehicle in self.vehicles:
            sa(str(vehicle))
            for tour in vehicle.tour:
                sa(str(tour))
        return "\n".join(s)


class Vehicle(object):
    id = count(1)

    def __init__(self, start, capacity, endurance, return_to_start=True):
        """
        :param start: Start location
        :param capacity: Vehicle capacity
        :param endurance: Time vehicle is available.
        """
        self.uid = next(Vehicle.id)
        self.start = start
        self.capacity = capacity
        self.used_capacity = 0
        self.endurance = endurance
        self.used_endurance = 0
        self._must_return_to_start = return_to_start
        load = Load(start, end=start, size=0, open=0, close=self.endurance, handling_time=0)
        self.tour = [load]
        self.pd = None  # place holder from ProblemDefinition (pd)

    def __str__(self):
        return f"Vehicle {self.uid}:"

    def tour_length(self):
        start = self.tour[0]
        t = 0
        for end in self.tour[1:]:
            load = end.handling_time
            unload = end.handling_time
            travel = self.pd.am[start.to_loc][end.to_loc]
            t += load + unload + travel
        return t

    def evaluate(self):
        if self.pd is None:
            raise AttributeError("forgot to add the vehicle? Use: cvrptw.add(vehicle)")
        assert isinstance(self.pd, CVRPTW)
        # 1. make a shortlist:
        ranking = [load for load in self.pd.loads if load.vehicle is None]

        while ranking:
            # 2. rank the shortlisted loads
            last_load = self.tour[-1]
            tour_length = self.tour_length()
            used_capacity = sum([l.size for l in self.tour])

            new_rank = []
            add = new_rank.append
            for load in ranking:
                assert isinstance(load, Load)
                if load.vehicle is not None:
                    continue
                if load.size + used_capacity > self.capacity:
                    continue
                leg = self.pd.am[last_load.to_loc][load.to_loc]
                if tour_length + leg > self.endurance:
                    continue
                if self._must_return_to_start:
                    home_leg = self.pd.am[load.to_loc][self.start]
                    if tour_length + leg + home_leg > self.endurance:
                        continue

                add((load.open, load.close, leg, load))

            if new_rank:
                new_rank.sort()
                ranking = [load for o, c, d, load in new_rank]
                load = ranking.pop(0)  # 3. pop top ranking load
                load.vehicle = self
                self.tour.append(load)
            else:
                break

        if self._must_return_to_start:
            self.create_home_trip()

    def create_home_trip(self):
        """ adds return trip to start after end of search. """
        self.tour.append(self.tour[0])


class Load(object):
    id = count(1)

    def __init__(self, start, end, size, open, close, handling_time):
        """
        :param start: start location
        :param end: end location
        :param size: Size (relative to vehicle capacity)
        :param open: Time window opening
        :param close: Time window closing
        :param time: Time to load/unload
        """
        self.uid = next(Load.id)
        self.from_loc = start
        self.to_loc = end
        self.size = size
        self.open = open
        self.close = close
        self.vehicle = None
        self.handling_time = handling_time

    def __str__(self):
        return f"Load {self.uid}: from {self.from_loc} to {self.to_loc}"

    def __lt__(self, other):
        return self.open < other.open

    def __le__(self, other):
        return self.open <= other.open
