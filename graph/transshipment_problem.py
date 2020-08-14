from time import process_time
from itertools import permutations, product

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

from graph.finite_state_machine import FiniteStateMachine


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
    return [(path[i], path[i+1]) for i in range(len(path)-1)]


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
        _, p = g.breadth_first_search(A, start)  # path back to start.
        if p:
            return [start] + p
    return []


def small_puzzle_resolve(graph, loads):
    """ calculates the solution to the transshipment problem."""
    assert isinstance(graph, Graph)
    assert isinstance(loads, dict)
    for v in loads.values():
        assert isinstance(v, list)
        assert all(i in graph.nodes() for i in v)

    # 0) where is the conflict?
    # 1) where are the free spaces?
    # 2) can detour into free space solve routing conflicts
    # 3) can an entity just wait?
    # 4) if all items go to N destinations, can a clockwise or counter clockwise
    #    preference solve the routing problem for good?
    # 5) Is thee a max flow that resolves the problem? If not, then there is
    #    probably a central bottleneck in the system...
    # 6) If all the quick rudimentary methods don't solve the problem, then engage the puzzle solve mode.

    initial_state = tuple(((load_id, route[0]) for load_id, route in loads.items()))
    final_state = tuple(((load_id, route[-1]) for load_id, route in loads.items()))
    # while no conflict: progress forward as given by paths.
    longest_path = 0
    for load_id, path in loads.items():
        if len(path) > 1:
            if not graph.has_path(path):
                _, path = graph.shortest_path(path[0], path[-1])
                loads[load_id] = path
        longest_path = max(longest_path, len(path))
    #
    # step = 0
    # pre_collision_step = None
    #
    # for step in range(longest_path):
    #     if pre_collision_step is not None:
    #         break
    #
    #     locations = set()
    #     for path in loads.values():
    #         if step + 1 < len(path):
    #             location = tuple(sorted(path[step:step+2]))
    #         else:
    #             location = path[-1]
    #         if location in locations:
    #             pre_collision_step = step
    #             break  # step where the conflict is detected.
    #             # now start backtracking.
    #         else:
    #             locations.add(location)
    #
    # pre_collision_step = max(0, step-1)  # step before collision.
    #
    # initial_state = []
    # for load_id, route in loads.items():
    #     if len(route) < pre_collision_step:
    #         initial_state.append((load_id, route[-1]))
    #     else:
    #         initial_state.append((load_id, route[pre_collision_step]))
    # initial_state = tuple(initial_state)

    # can the problem be solved by having one set of units wait?
    movements = Graph()

    # at conflict: reverse until solvable.
    states = [initial_state]
    while states:
        state = states.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:
            if final_state in movements:
                break
            options = (e for s, e, d in graph.edges(from_node=location) if e not in occupied)
            for option in options:
                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)

                if new_state in movements:  # abandon branch
                    continue
                else:
                    # assign affinity towards things moving in same direction.
                    movements.add_edge(state, new_state, 1)
                    states.append(new_state)

                if final_state in movements:
                    states.clear()
                    break

    if final_state not in movements:
        raise Exception("No solution found")

    steps, best_path = movements.shortest_path(initial_state, final_state)
    moves = path_to_moves(best_path)
    return moves


# ----------------- #
# The main solver   #
# ----------------- #
def resolve(graph, loads):
    """ an ensemble solver for the routing problem."""
    check_user_input(graph, loads)

    moves = None
    for method in methods:
        try:
            moves = method(graph, loads)
        except TimeoutError:
            pass
        if moves:
            return moves
    return moves


# helpers
def check_user_input(graph, loads):
    """ checks the user inputs to be valid. """
    assert isinstance(graph, Graph)
    assert isinstance(loads, dict)
    for v in loads.values():
        assert isinstance(v, list)
        assert all(i in graph.nodes() for i in v)

    for load_id, path in loads.items():
        if len(path) > 1:
            if not graph.has_path(path):
                _, path = graph.shortest_path(path[0], path[-1])
                loads[load_id] = path


def path_to_moves(path):
    """translate best path into motion sequence."""
    moves = []
    s1 = path[0]
    for s2 in path[1:]:
        for p1, p2 in zip(s1, s2):
            if p1 != p2:
                moves.append(
                    {
                        p1[0]: (  # load id.
                            p1[1],  # location 1.
                            p2[1]   # location 2.
                        )
                    })
        s1 = s2
    return moves


def clockwise_turn(rows):
    """ performs a clock wise turn of items in a grid """
    rows2 = [[v for v in r] for r in rows]
    rows2[0][1:] = [v for v in rows[0][:-1]]  # first row,
    rows2[-1][:-1] = [v for v in rows[-1][1:]]  # last row
    for ix, row in enumerate(rows[1:]):  # left
        rows2[ix][0] = row[0]
    for ix, row in enumerate(rows[:-1]):  # right
        rows2[ix + 1][-1] = row[-1]
    return rows2


def counterclockwise_turn(rows):
    """ performs a counter clock wise turn of items in a grid """
    rows2 = [[v for v in r] for r in rows]
    rows2[0][:-1] = [v for v in rows[0][1:]]  # first row
    rows2[-1][1:] = [v for v in rows[-1][:-1]]  # last row
    for ix, row in enumerate(rows[:-1]):  # left side
        rows2[ix + 1][0] = row[0]
    for ix, row in enumerate(rows[1:]):  # right side
        rows2[ix][-1] = row[-1]
    return rows2


def change(items, c, d):
    if c == 1:
        square = [v for v in items[0][:2]], [v for v in items[1][:2]]
    elif c == 2:
        square = items
    else:
        square = [v for v in items[0][1:]], [v for v in items[1][1:]]

    if d == 'cw':
        square = clockwise_turn(square)
    else:
        square = counterclockwise_turn(square)

    if c == 1:
        items[0][:2] = [v for v in square[0]]
        items[1][:2] = [v for v in square[1]]
    elif c == 2:
        items = [[v for v in row] for row in square]
    else:
        items[0][1:] = [v for v in square[0]]
        items[1][1:] = [v for v in square[1]]
    return items


def resolve2x3(initial_state, desired_state):
    def state(items):
        return "".join(str(i) for i in items)

    items = initial_state
    all_options = list(list(i) for i in permutations(items, 6))

    options = [1, 2, 3]
    directs = ['cw', 'ccw']

    g = FiniteStateMachine()

    while all_options:
        items = all_options.pop()
        current_state = state(items)

        for c, d in list(product(*[options, directs])):
            new_items = change([items[:3], items[3:]], c, d)
            new_state = state(new_items[0][:] + new_items[1][:])
            g.add_transition(state_1=current_state, action=(c, d), state_2=new_state)

    d, p = g.states.shortest_path(initial_state, desired_state)
    return p, g


def find_trains(graph, loads):
    """ reads the paths and determines all trains (each linear cluster of loads)"""
    check_user_input(graph, loads)
    # check that the order of movements are constant.
    return [(1, 2, 3), (4, 5, 6, 7)]


def first_car(train, loads):
    """ determines the end of the train that is leading. """
    for end in [0, -1]:
        first_car = train[end]
        route = loads[first_car]
        if len(route) == 1:  # this train isn't moving.
            return None

        locations = {loads[load_id][0] for load_id in train if load_id != first_car}

        if any(i in locations for i in route):
            continue  # this is the wrong end of the train.
        else:
            return first_car
    return None


# solution methods.
def train_resolve(graph, loads):
    """ A greedy algorithm that finds and removes trains from the network """
    check_user_input(graph, loads)

    # 1. who can move?
    # evidently only the start and end of a train, because the middle
    # is locked between the two ends.
    # So, if a train moves in a direction, it means that all wagons follow
    # the leading wagon (until the paths potentially divide).
    # The leading wagon must then "know" how long the train is, so that the
    # solution landscape can be reduce to routes that are passable for whole
    # trains.

    # 2. as the trains are known, the next question is then how to reduce the
    # graph to "train-friendly" routes.

    # 3. Let's start with finding the biggest train, then find a route for it.
    trains = find_trains(graph, loads)
    assert isinstance(trains, list)
    assert all(isinstance(i, tuple) for i in trains)
    trains.sort(key=lambda x: len(x))
    # 4. Once the biggest train is out of the way, we look for where we can store
    #    obstacles in the mean time.

    while trains:
        train = trains.pop(0)
        first = first_car(train, loads)
        last = train[-1] if first == train[0] else train[0]
        path = set(loads[first]).union(loads[last])
        obstacles = [load_id for load_id, route in loads.items() if load_id not in train and path.intersection(set(route))]
        # can obstacles move out of the way of first?
        free_locations = [n for n in graph.nodes() if n not in path]
        if len(free_locations) < len([lid for lid in obstacles if lid not in train]):
            return None  # solving using this method not possible.
        #todo: make a plan that moves the obstacles out of the way.
        #      get rid of train 1.
        #      reduce the graph, so that train 1 doesn't have to move again.
    # 5. Repeat from 3. Until all trains are removed.
    pass


def dfs_resolve(graph, loads, time_limit_ms=10000):
    """calculates the solution to the transshipment problem."""
    assert isinstance(time_limit_ms, (float, int))

    initial_state = tuple(((load_id, route[0]) for load_id, route in loads.items()))
    final_state = tuple(((load_id, route[-1]) for load_id, route in loads.items()))
    movements = Graph()

    start = process_time()

    states = [initial_state]
    while states:
        if process_time() - start > (time_limit_ms / 1000):
            raise TimeoutError(f"No solution found in {time_limit_ms}ms")

        state = states.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:
            if final_state in movements:
                break
            options = (e for s, e, d in graph.edges(from_node=location) if e not in occupied)
            for option in options:
                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)

                if new_state in movements:  # abandon branch
                    continue
                else:
                    movements.add_edge(state, new_state, 1)
                    states.append(new_state)

                if final_state in movements:
                    states.clear()
                    break

    if final_state not in movements:
        raise Exception("No solution found")

    steps, best_path = movements.shortest_path(initial_state, final_state)
    moves = path_to_moves(best_path)
    return moves


# collection of solution methods for the routing problem.
# insert, delete, append or substitute with your own methods as required.
methods = [
    # train_resolve,
    dfs_resolve
]

