from graph import Graph
from itertools import count
from collections import deque
import itertools

__description__ = """

This is the slight more advanced example with time windows.

Transshipment problems form a subgroup of transportation problems, where 
transshipment is allowed. In transshipment, transportation may or must go 
through intermediate nodes, possibly changing modes of transport.

Transshipment or Transhipment is the shipment of goods or containers to an 
intermediate destination, and then from there to yet another destination. One 
possible reason is to change the means of transport during the journey (for 
example from ship transport to road transport), known as transloading. Another 
reason is to combine small shipments into a large shipment (consolidation), 
dividing the large shipment at the other end (deconsolidation). Transshipment 
usually takes place in transport hubs. Much international transshipment also 
takes place in designated customs areas, thus avoiding the need for customs 
checks or duties, otherwise a major hindrance for efficient transport.

[1](https://en.wikipedia.org/wiki/Transshipment_problem)

Prologue: Scheduling!
------------------------------------------------------------------------------
The proposal is to treat the jobs as actors instead of objects.

The N-machine scheduling problem is thereby a problem defined as:

Determine the solution with minimum total slack time (from start to end)
subject to the constraints:
1. total ordering given by completeness of each order as a set.
   (All items required in an order must be packed and shipped together)
2. The capacity constraints of each machine.
3. The route of each job through the network of machines.

The emphasis is therefore on the proposition that jobs need to search for
paths to reach their required destination within a time-window and book
their space on the machines as required.

Think of this a person attempting to travel from Oxford University to MIT,
Boston: The trip contains a number of route options:
    1. fly from London Heathrow or from Birmingham to Boston Logan
    2. drive to the airport or take train, taxi, ...
    3. rent a car, take a taxi, ... at the destination airport.

A job is thereby a path through a number of way points.
The job will face resource constraints (travel time, time windows)

Likewise the machinery involved will face a number of constraints, and if
it's completely utilised, it will not care one bit whether it handles one
job or another. A taxi driver however will care if one job provides the
opportunity for a return trip whilst another doesn't. 

However attempting to let the Taxi determine the optimal schedule, in
contrast to the travel, seems intuitively like a bad a idea. Yet that is
what many scheduling methods attempt to do.

Conclusion: Passengers are actors, not objects.

Finally some vehicles, vessels, etc (generally transports) have constraints with
regard to loading and unloading. A container ship is an obvious example:

To get to a container lower in the ship, the containers above must be removed.
For a bus, train or other public transport this contraints may not apply.

"""

class Agent(object):
    uuid = count(0)

    def __init__(self, network):
        self.uuid = next(Agent.uuid)
        self.network = network
        self.trips = []

    def add_trip(self, trip):
        assert isinstance(trip, PassengerTrip)
        self.trips.append(trip)

    def schedule(self):
        """
        schedules self.
        :return: None
        """
        pass

    def schedule_complete(self):
        """
        :return: boolean
        """
        return all(t.has_valid_travel_plan() for t in self.trips)


class Passenger(Agent):
    def __init__(self, network, trips=None):
        """
        :param network: graph of the network 
        :param trips: list of trip requirements as tuples with
                      (start, end, available, deadline)
        """
        super().__init__(network=network)
        if trips is not None:
            self._add_trips(trips)

    def schedule(self):
        pass

    def _add_trips(self, list_of_trips):
        """
        Helper for adding trips.
        :param list_of_trips: list of tuples. Each tuple must comply to
                              Trip.__init__()
        """
        assert isinstance(list_of_trips, list)
        for trip in list_of_trips:
            self.trips.append(PassengerTrip(network=self.network, *trip))


class InterSection(object):
    def __init__(self, capacity=float('inf')):
        self.capacity = capacity


class Transporter(Agent):
    FILO = 1
    FIFO = 2

    def __init__(self, network, start, stops=None,
                 capacity=float('inf'), order=None,
                 loading_time=None):
        """
        :param network: graph of the network
        :param start: a (location) node in the network.
        :param stops: Set of nodes where the transporter stops.
                      if `stops=None` then the transporter can stop at all nodes.
        :param capacity: the number of passengers that can board.
        :param order: the dis-/embarkation order for passengers. Options:
                      None: No ordering.
                      Shuttle.FILO: 1
                      Shuttle.FIFO: 2
        """
        super().__init__(network=network)

        assert start in network
        self._current_location = start
        assert isinstance(stops, set)
        assert all([n in network for n in stops])
        if stops is None:
            stops = set(network.nodes())
        self.stops = stops

        self.trips = []
        self.capacity = capacity
        self._cargo = []  # used to determine FIFO/FILO.

        if order not in (None, Transporter.FILO, Transporter.FIFO):
            raise ValueError("Unexpected value for order={}".format(order))
        self.order = order

        if loading_time is None:
            loading_time = 0
        self.loading_time = loading_time

        self.itinerary = []

    def schedule(self):
        """
        :param trips: (optional) list of trips to schedule.
        :return: schedule
        """
        pass


class WayPoint(object):
    """
    A point that must be passed between open and close time.

    This should be used, for example when a strict arrival sequence
    is expected.
    """
    __slots__ = ['open', 'close', 'location']
    def __init__(self, location, open=None, close=None):
        """
        :param open: time the waypoint opens.
        :param close: time the waypoint closes.
        :param location: location of the waypoint.
        """
        if open is None:
            open = 0
        self.open = open
        if close is None:
            close = float('inf')
        self.close = close
        if close < open:
            raise ValueError("close < open: {} < {}".format(close, open))
        self.location = location


class Trip(object):
    def __init__(self, network, waypoints):
        assert isinstance(network, Graph)
        self.network = network
        assert len(waypoints) >= 2
        assert all(isinstance(wp, WayPoint) for wp in waypoints)
        self.waypoints = waypoints

    def check_route(self):
        """
        check that the route is valid.
        :return:
        """
        pass # TODO: Add intersection capacity to the scheduling method.


class PassengerTrip(Trip):
    def __init__(self, network, waypoints):
        """
        :param network: graph of the network 
        :param start: start of the journey
        :param end: end of the journey
        :param a: _available_ to travel from this time.
        :param d: _deadline_ for arrival at destination
        :param b: _begin_ journey
        :param c: _complete_ journey

        Illustrated
        |------------------------------>  time
        |     A <----------------> D      available from - to deadline
        |               B <-----> C       valid schedule

        :param waypoints: (optional) path for the journey (if not shortest path)

        """
        super().__init__(network, waypoints=waypoints)

        self.routes = Graph()
        self.routes.add_node(node_id=start, obj=WayPoint(self, start, self.available))
        self.routes.add_node(node_id=end, obj=WayPoint(self, end, self.deadline))

    def has_valid_travel_plan(self):
        """
        :return: boolean, True if a valid travel plan can be constructed.
        """
        # 1. for each path amongst all possible paths (sorted by duration)
        # 2.       if the path has a valid sequence of carriers,
        #                keep it.
        pass

    def can_travel_using(self, transport):
        """
        Determines if a transport option is useful for the trip and stores
        the option in self.routes (a Graph)

        Hereby 3 options like:

                 (start)
                  board   disembark
                    V      V
        T1: |-------+------|---------> (fast train)
        T2:         |-+-+--|------->   (local train)
        T3:                +-----|     (taxi)
                               (end)

        Become a graph like:

        start-----> T1 --> T3 --- end
             \-----T2 --->/

        From this point we can use the shortest path algorithm to determine
        the fastest route.

        :param transport: instance of class Transporter
        :return: None
        """
        assert isinstance(transport, Transporter)

        path = [p for p in self.path if p in transport.stops]
        if len(path) < 2:  # there may be a shared stop, but no travel option.
            return
        self.routes.add_node(node_id=transport.uuid, obj=path)
        length = self.network.distance_from_path(path)

        start = self.routes.node('start')



        if self.start in path:
            self.routes.add_edge(self.start)


        self.routes.add_edge(A, B, length)


class TransporterTrip(object):
    def __init__(self, network, waypoints):
        """
        :param network: graph of the network
        :param start: start of the journey
        :param end: end of the journey
        :param a: _available_ to travel from this time.
        :param d: _deadline_ for arrival at destination
        :param b: _begin_ journey
        :param c: _complete_ journey

        Illustrated
        |------------------------------>  time
        |     A <----------------> D      available from - to deadline
        |               B <-----> C       valid schedule

        :param path: (optional) path for the journey (if not shortest path)

        """
        super().__init__(network, waypoints)


class Event(object):
    def __init__(self, start, end):
        """
        :param start: start time
        :param end: end time
        """
        self.start = start
        self.end = end


class Move(Event):
    def __init__(self, start, end, A, B):
        """
        :param start: start time
        :param end: end time
        :param A: origin
        :param B: destination
        """
        super().__init__(start, end)
        self.a = A
        self.b = B


class Load(Event):
    def __init__(self, start, end, what):
        """

        :param start:
        :param end:
        :param what:
        """
        super().__init__(start, end)
        self.what = what


class UnLoad(Event):
    def __init__(self, start, end, what):
        super().__init__(start, end)
        self.what = what


def schedule_all_trips(transports, passengers):
    """
    Iterative over the transports and passengesr until all have a feasible
    schedule .

    1. Each device loop through the jobs:
        select all relevant jobs.
        create itinerary items on the jobs.
        when all itineraries are complete, return the schedule.

    """
    assert isinstance(transports, list)
    assert isinstance(passengers, list)

    # create initial relationships.
    for passenger in passengers:
        assert isinstance(passenger, Passenger)
        for trip in passenger.trips:
            assert isinstance(trip, PassengerTrip)
            for transport_option in transports:
                trip.can_travel_using(transport_option)

    incomplete_schedules = deque(transports + passengers)

    while incomplete_schedules:
        entity = incomplete_schedules.pop()
        pass  # TODO:

        if not entity.schedule_complete():
            incomplete_schedules.append(entity)


    # open_schedules = deque(transports)
    # while open_schedules:
    #     for transport in transports:
    #         assert isinstance(transport, Transporter)
    #         # check/find passengers.
    #         trips = []
    #         for passenger in passengers:
    #             assert isinstance(passenger, Passenger)
    #             trips.extend(passenger.travels_through(transport.stops))
    #
    #         # find a valid sequence.
    #         transport.schedule(trips)


def find_options(passenger, transports):
    assert isinstance(passenger, Passenger)
    assert isinstance(transports, list)
    assert all(isinstance(i, Transporter) for i in transports)
    for trip in passenger.trips:
        for transport in transports:
            pass



def schedule(graph, start, jobs):
    """
    The best possible path is a circuit.
    First we'll find all circuits and attempt to remove them from
    the equation.

    Once no more circuits can be found, the best solution is to look for
    alternative combinations that provide improvement.

    :return:
    """
    new_schedule = []
    jobs_to_plan = jobs[:]
    while jobs_to_plan:
        circuit_path = find_perfect_circuit(graph=graph, start=start, jobs=jobs_to_plan)
        if circuit_path:
            job_sequence = jobs_from_path(circuit_path)
        else:  # circuit not possible.
            shortest_path = []
            shortest_distance = float('inf')
            for perm in itertools.permutations(jobs_to_plan, len(jobs_to_plan)):
                path = path_from_schedule(jobs=perm, start=start)
                distance = graph.distance_from_path(path)
                if distance < shortest_distance:
                    shortest_distance = distance
                    shortest_path = path
            job_sequence = jobs_from_path(shortest_path)
        # remove planned jobs from options:
        for job in job_sequence:
            if job in jobs_to_plan:
                jobs_to_plan.remove(job)
                new_schedule.append(job)
    return new_schedule


def jobs_from_path(path):
    """ helper for finding jobs from path"""
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def path_from_schedule(jobs, start):
    """ The evaluation is based on building the travel path.
    For example in the network A,B,C with 4 trips as:
        1 (A,B), 2 (A,C), 3 (B,A), 4 (C,A)
    which have the travel path: [A,B,A,C,B,A,C,A]

    The shortest path for these jobs is: [A,C,A,B,A] which uses the order:
        2 (A,C), 4 (C,A), 1 (A,B), 3(B,A)
    """
    path = [start]
    for A, B in jobs:
        if A != path[-1]:
            path.append(A)
        path.append(B)
    return path


def find_perfect_circuit(graph, start, jobs):
    """ A perfect circuit is a path that starts and ends at the same place
    and where every movement includes a job.

    :param: start: starting location.
    :param: jobs: list of movements [(A1,B1), (A2,B2,) ....]
    :return path [A1,B1, ..., A1]
    """
    g = Graph()
    for A, B in jobs:
        try:
            g[A][B]
        except KeyError:
            d, p = graph.shortest_path(A, B)
            g.add_edge(A, B, d)

    new_starts = [B for A, B in jobs if A == start]
    for A in new_starts:
        _, p = g.breadth_first_search(A, start)  # path back to start.
        if p:
            return [start] + p
    return []




