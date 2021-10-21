from graph import Graph
import itertools

__description__ = """
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
"""


def clondike_transshipment_problem():
    """
    A deep gold mining operation is running at full speed.

    Problem:
    What schedule guarantees an optimal throughput given unpredictable output
    from the mines and unpredictable needs for mining equipment?

    Constraints:
    The gold mine has a surface depot which connects the 4 mining levels using
    an elevator running in the main vertical mine shaft.
    At each level there is a narrow gage electric train which can pull rail cars
    between the lift and side tracks where excavation occurs.
    Loading minerals and unloading equipment takes time, so the best way to
    avoid obstructing the busy lift is by moving the rail cars directly into
    the lift. However as mining equipment and valuable minerals are very heavy,
    the lift can only move one load at the time.
    Similarly the best way to avoid obstructing the railway at each level is by
    by moving the narrow gage rail cars onto side tracks that are available at
    each horizontal mine entry.
    Schematic of the mine:
        Mining shaft lift to surface depot.
        ^
        |
        +- level 1 <------------------->
        |              1  2  3  ... N  (horizontal mine shafts)
        |
        +- level 2 <------------------->
        |              1  2  3  ... N
        |
        +- level 3 <------------------->
        |              1  2  3  ... N
        |
        +- level 4 <------------------->
        |              1  2  3  ... N
        |
        +- level N
    In the intersection between the mining shaft lift and the electric
    locomotive there is limited space to perform any exchange, and because of
    constraints of the equipment, the order of delivery is strict:
       ^ towards surface.
       |
       |            (railway switch)
       |                |
    [ from lift ] ---->[S]<-->[ electric locomotive ]<---> into the mine.
    [ onto lift ] <-----|
       |
       |
       v into the mine.
    """
    paths = [
        ("Surface", "L-1", 1),
        ("L-1", "L-2", 1),
        ("L-2", "L-3", 1),
        ("L-3", "L-4", 1),
        ("L-1", "L-1-1", 1),
        ("L-2", "L-2-1", 1),
        ("L-3", "L-3-1", 1),
        ("L-4", "L-4-1", 1),
    ]

    for level in [1, 2, 3, 4]:  # adding stops for the narrow gage trains in the levels.
        paths.append(("L-{}".format(level), "L-{}-1".format(level), 1), )
        for dig in [1, 2, 3, 4, 5, 6]:
            paths.append(("L-{}-{}".format(level, dig), "L-{}-{}".format(level, dig + 1), 1))

    paths.extend([(n2, n1, d) for n1, n2, d in paths])  # adding the reverse path.
    g = Graph(from_list=paths)
    return g


class Train(object):
    def __init__(self, rail_network, start_location, access):
        """
        :param rail_network: the while rail network as a Graph.
        :param start_location: a node in the network.
        :param access: Set of nodes to which this train has access.
        """
        assert isinstance(rail_network, Graph)
        self._rail_network = rail_network
        assert start_location in self._rail_network.nodes()
        self._current_location = start_location
        assert isinstance(access, set)
        assert all([n in self._rail_network for n in access])
        self._access_nodes = access

        self._schedule = []

    def schedule(self, jobs=None):
        """
        Initialise the solution using shortest jobs first.
        Then improve the solution using combinatorics until improvement is zero
        param: jobs, (optional) list of jobs
        returns: list of jobs in scheduled order.
        """
        if jobs is None:
            return self._schedule
        assert isinstance(jobs, list)
        new_jobs = find(rail_network=self._rail_network, stops=self._access_nodes, jobs=jobs)
        self._schedule = schedule(graph=self._rail_network, start=self._current_location, jobs=new_jobs)
        return self._schedule


def schedule_rail_system(rail_network, trains, jobs):
    """
    Iterative over the trains until a feasible schedule is found.
    1. Each device loop through the jobs:
        select all relevant jobs.
        create itinerary items on the jobs.
        when all itineraries are complete, return the schedule.
    """
    assert isinstance(rail_network, Graph)
    assert isinstance(trains, list)
    assert all(isinstance(t, Train) for t in trains)
    assert isinstance(jobs, list)

    for train in trains:
        assert isinstance(train, Train)
        train.schedule(jobs)


def find(rail_network, stops, jobs):
    """
    Finds the route sections that the jobs need to travel from/to (stops)
    in the rail network.
    :param rail_network: class Graph
    :param stops: set of train stops
    :param jobs: list of jobs.
    :return: sub jobs.
    """
    sub_jobs = []
    for A, B in jobs:
        if A not in rail_network:
            raise ValueError("{} not in rail network".format(A))
        if B not in rail_network:
            raise ValueError("{} not in rail network".format(B))

        _, path = rail_network.shortest_path(A, B)
        part_route = [p for p in path if p in stops]

        if not rail_network.has_path(part_route):
            raise ValueError("Can't find path for {}".format(part_route))

        new_a, new_b = part_route[0], part_route[-1]
        sub_jobs.append((new_a, new_b))
    return sub_jobs


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
            g.edge(A, B)
        except KeyError:
            d, p = graph.shortest_path(A, B)
            g.add_edge(A, B, d)

    new_starts = [B for A, B in jobs if A == start]
    for A in new_starts:
        if A in g:
            p = g.breadth_first_search(A, start)  # path back to start.
            if p:
                return [start] + p
    return []

