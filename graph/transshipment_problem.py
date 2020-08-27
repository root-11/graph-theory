from time import process_time
from itertools import permutations, product

TIMEOUT = 10_000_000_000  # ms.

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
        p = g.breadth_first_search(A, start)  # path back to start.
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
            print(f"{method.__name__} found a solution.")
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
    """ translate best path into motion sequence.
    :param path: list of tuples with [(load id, location), ... ]
    """
    moves = []
    s1 = path[0]
    for s2 in path[1:]:
        for p1, p2 in zip(s1, s2):
            if p1 != p2:
                moves.append({p1[0]: (p1[1], p2[1])})  # {load id: (location1, location2)}
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


def merge(paths):
    """
    :param paths: [list of nodes]
    :return: path as list of nodes
    """
    g = Graph()
    for path in paths:
        a = path[0]
        for b in path[1:]:
            v = g.edge(a, b)
            if v is None:
                g.add_edge(a, b, 1)
            else:
                g.add_edge(a, b, v + 1)
            a = b
    start, end = g.nodes(in_degree=0), g.nodes(out_degree=0)
    d, p = g.shortest_path(start[0], end[0])
    return p


def test_merge():
    common_path = merge([[1, 2, 3, 4, 5, 9, 10, 11, 12],
                         [2, 3, 4, 5, 9, 10, 11, 12, 13],
                         [3, 4, 5, 9, 10, 11, 12, 13, 14]])
    assert common_path == [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14], common_path


def find_trains(graph, loads):
    """ reads the paths and determines all trains (each linear cluster of loads)"""
    check_user_input(graph, loads)
    # check that the order of movements are constant.
    initial_locations = [v[0] for v in loads.values()]
    g = graph.subgraph_from_nodes(initial_locations)
    train_locations = g.components()

    reverse_load = {v[0]: k for k, v in loads.items()}
    trains = [[reverse_load[loc] for loc in train_locs] for train_locs in train_locations]

    train_routes = {}
    for z in trains[:]:  # for each train get the route
        routes = {lid: route for lid, route in loads.items() if lid in z}
        common_path = merge(routes.values())
        train_routes[tuple(z)] = common_path  # sort the tuple in order of travel: First element must move first.
    return train_routes


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


class TimeLocationMap(object):
    def __init__(self, loads):
        assert isinstance(loads, dict)
        assert all([isinstance(i, list) for i in loads.values()])
        self.time_location = tl = {}  # time-location
        for load_id, route in loads.items():
            for step, location in enumerate(route):
                step_loc = (step, location)
                content = tl.get(step_loc, None)
                if not content:
                    tl[step_loc] = [load_id]
                else:
                    tl[step_loc].append(load_id)

    def conflicts(self):
        for k, v in self.time_location.items():
            if len(v) > 1:
                (step, location), loads = k, v
                yield step, location, loads


# solution methods.
def avoid_resolve(graph, loads):
    """ detects immediate conflicts and attempts to reroute """

    tl = TimeLocationMap(loads)

    loads2 = {k: v[:] for k, v in loads.items()}
    tl2 = TimeLocationMap(loads2)

    obstacles = set()
    for step, location, conflict_loads in tl.conflicts():
        for load in conflict_loads:  # should always be 2.
            old_route = loads[load]
            if len(old_route) <= 2:  # this load isn't moving.
                continue

            new_route = graph.avoids(start=old_route[0], end=old_route[-1], obstacles=[location] + list(obstacles))
            if not new_route:
                continue
            else:
                obstacles.update(set(old_route).difference(set(new_route)))
            # make a copy of the loads and check if it works.
            loads2[load] = new_route
            tl2 = TimeLocationMap(loads2)
            if list(tl2.conflicts()):
                continue

    if tl2.conflicts():
        return None

    if isinstance(loads2, dict):
        L = []
        steps = max([len(route) for route in loads2.values()])
        for step in range(steps):
            L.append(
                tuple((load_id, route[step]) if len(route) >= step else (load_id, route[-1])
                      for load_id, route in loads2.items())
            )
        moves = path_to_moves(L)
    else:
        moves = None

    return moves


def loop_resolve(graph, loads):
    """ The solution is some rotation of the current state"""
    pass


def train_resolve(graph, loads):
    """ A greedy algorithm that finds loads that can move to their final
    destination as a group. """
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
    # 4. Once the biggest train is out of the way, we look for where we can store
    #    obstacles in the mean time.
    trains = [(k, v) for k, v in trains.items()]
    trains.sort(key=lambda x: len(x[0]), reverse=True)

    loads2 = {}
    while trains:
        train, route = trains.pop(0)
        train_locations = [loads[t][0] for t in train]
        obstacles = [load_id for load_id, route in loads.items() if load_id not in train and set(route).intersection(set(route))]
        # can obstacles move out of the way of first?
        free_locations = [n for n in graph.nodes() if n not in route]
        if len(free_locations) < len([lid for lid in obstacles if lid not in train]):
            return None  # solving using this method not possible.
        # is it possible to take a train to a free location?
        paths = []
        for train2, route2 in trains:
            for lid in train2:
                for free_location in free_locations:
                    a, x = route2[0], free_location
                    x, b = free_location, route2[-1]
                    da, A = graph.shortest_path(a, x)
                    db, B = graph.shortest_path(x, b)
                    if da + db == float('inf'):
                        continue  # no path.
                    new_route = A + B[1:]

        exits = [n for n in graph.nodes()]

        #todo: make a plan that moves the obstacles out of the way.
        #      get rid of train 1.
        #      reduce the graph, so that train 1 doesn't have to move again.
    # 5. Repeat from 3. Until all trains are removed.
    pass


def bfs_resolve(graph, loads):
    """
    calculates the solution to the transshipment problem by
    constructing the solution space as a finite state machine
    and then finding the shortest path through the fsm from the
    initial state to the desired state.
    """
    initial_state = tuple(((load_id, route[0]) for load_id, route in loads.items()))
    final_state = tuple(((load_id, route[-1]) for load_id, route in loads.items()))
    movements = Graph()

    start = process_time()

    states = [initial_state]
    solved = False

    while not solved:
        if process_time() - start > (TIMEOUT / 1000):
            raise TimeoutError(f"No solution found in {TIMEOUT}ms")

        state = states.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:
            if solved: break
            options = (e for s, e, d in graph.edges(from_node=location) if e not in occupied)
            for option in options:
                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)

                if new_state in movements:  # abandon branch
                    continue
                else:
                    movements.add_edge(state, new_state, 1)
                    states.append(new_state)

                if final_state == new_state:
                    solved = True
                    break
        if not states:
            raise Exception("No solution found")

    steps, best_path = movements.shortest_path(initial_state, final_state)
    moves = path_to_moves(best_path)
    return moves


def dfs_resolve(graph, loads):
    """
    calculates the solution to the transshipment problem by
    search along a line of movements and backtracking when it
    no longer leads anywhere (DFS).
    """
    initial_state = tuple(((load_id, route[0]) for load_id, route in loads.items()))
    final_state = tuple(((load_id, route[-1]) for load_id, route in loads.items()))

    state = initial_state
    states = [initial_state]  # q
    path = []
    movements = Graph()
    visited = set()
    start = process_time()

    while states:
        if process_time() - start > (TIMEOUT / 1000):
            raise TimeoutError(f"No solution found in {TIMEOUT}ms")

        state = states.pop(0)  # n1
        visited.add(state)  # visited
        path.append(state)  # path
        if state == final_state:
            states.clear()  # return path  # exit if final state is found.
            break

        for new_state in new_states(graph, movements, state):  # for n2 in g.from(n1)
            if new_state in visited:  # if n2 in visited
                continue
            states.append(new_state)  # q.append(n2)
            break
        else:
            path.remove(state)  # path.remove(n1)
            while not states and path:  # while not q and path
                for new_state in new_states(graph, movements, state=path[-1]):  # for n2 in g.from_node(path[-1]):..
                    if new_state in visited:  # if n2 in visited
                        continue
                    states.append(new_state)
                    break
                else:
                    path = path[:-1]
    if state != final_state:
        return None  # <-- exit if not path was found.

    steps, best_path = movements.shortest_path(initial_state, final_state)
    moves = path_to_moves(best_path)
    return moves



def new_states(graph, movements, state):
        occupied = {i[1] for i in state}
        for load_id, location in state:
            options = (e for s, e, d in graph.edges(from_node=location) if e not in occupied)
            for option in options:
                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)
                if new_state in movements:
                    continue
                movements.add_edge(state, new_state, 1)
                yield new_state


class Load(object):
    def __init__(self, id, path, schedule):
        self.id = id
        assert isinstance(path, list)
        self.path = path
        self.schedule = schedule

    def __repr__(self):
        return f"Load({self.id}): {self.path}"

    def avoid(self, step, location, loads):
        g = self.schedule.graph
        assert isinstance(g, Graph)
        s, e = self.path[step - 1], self.path[step + 1]

        current_path = new_path = self.path[:]
        if s == location == e:  # only option is to step out of the way (and `avoid` wont work).
            for a, b, d in g.edges(from_node=location):  # for each neighbour,
                _, p = g.shortest_path(b, location)  # what's the shortest path back.
                alt_path = self.path[:step] + p[:-1] + self.path[step + 1:]  # what is the alternative route?
                self.path = alt_path[:]

                remaining_conflicts = list(self.schedule.conflicts(loads))  # did the conflict vanish?
                if remaining_conflicts:
                    for s, l, loads_ in remaining_conflicts:
                        if all([s == step, l == location, not set([L.id for L in loads]).difference([L.id for L in loads_])]):
                            self.path = current_path[:]  # change didn't help. return to pre-change.
                            break
                    else:
                        return  # problem solved.
                else:
                    return  # problem solved.

        elif s != location != e:  # change the route.
            detour = g.avoids(s, e, [location])
            new_path = self.path[:step] + detour + self.path[step + 1:]

        else:  # wait to see if others move.
            new_path = self.path[:step] + [self.path[step - 1]] + self.path[step + 1:]

        assert self.path[0] == new_path[0] and self.path[-1] == new_path[-1], "bad route!"
        self.path = new_path


class Schedule(object):
    def __init__(self, graph, loads=None):
        self.graph = graph
        self.loads = {}
        if loads is not None:
            for k, v in loads.items():
                self.loads[k] = Load(k, v, self)
            duration = self.duration()
            for load in self.loads.values():
                if len(load.path) < duration:
                    dx = duration - len(load.path)
                    load.path = load.path + [load.path[-1]] * dx
        assert len({len(v.path) for v in self.loads.values()}) == 1, "routes have different lengths."

    def duration(self):
        return max(len(load.path) for load in self.loads.values())

    def conflicts(self, loads=None):
        if loads is None:
            loads = self.loads.values()

        d = {}
        # for load in loads:
        #     assert isinstance(load, Load)
        #     for t, loc in enumerate(load.path):
        #         tloc = (t, loc)
        #         content = d.get(tloc, None)
        #         if not content:
        #             d[tloc] = [load]
        #         else:
        #             d[tloc].append(load)

        # Two opposite time steps must be detected.
        durations = {len(load.path) for load in loads}
        if len(durations) != 1:  # "routes have different lengths."
            for load in loads:
                if len(load.path) < max(durations):
                    load.path = load.path + [load.path[-1]] * (max(durations) - len(load.path))
        duration = max(durations)

        for step in range(duration-1):
            for load in loads:
                swops = {(L2.path[step], L2.path[step + 1]):L2 for L2 in loads if L2 is not load}
                # swops are cases where the load change places by passing through each other
                # (impossible move): Load A: 1 --> 2, whilst Load B: 2 --> 1.

                swop_value = (load.path[step+1], load.path[step])
                if swop_value not in swops:
                    continue
                L2 = swops[swop_value]

                tloc = (step, load.path[step])

                conflict = d.get(tloc, None)

                if not conflict:
                    d[tloc] = [load, L2]
                else:
                    if load not in d[tloc]:
                        d[tloc].append(load)
                    if L2 not in d[tloc]:
                        d[tloc].append(L2)

        for step in range(duration):
            for load in loads:
                if load.path[step] in {L2.path[step] for L2 in loads if L2 is not load}:
                    tloc = (step, load.path[step])
                    conflict = d.get(tloc, None)
                    if not conflict:
                        d[tloc] = [load]
                    else:
                        if load not in d[tloc]:
                            d[tloc].append(load)
                else:
                    pass  # everything is ok.

        for k, v in sorted(d.items()):
            if len(v) > 1:
                (step, loc), loads = k, v
                yield step, loc, loads


def action_resolve(graph, loads):
    """calculates the solution to the transshipment problem."""
    initial_state = tuple(((load_id, route[0]) for load_id, route in loads.items()))
    final_state = tuple(((load_id, route[-1]) for load_id, route in loads.items()))

    start = process_time()

    s = Schedule(graph, loads)
    conflicts = list(s.conflicts())

    while conflicts:
        if process_time() - start > (TIMEOUT / 1000):
            raise TimeoutError(f"No solution found in {TIMEOUT}ms")

        conflict = conflicts.pop(0)
        step, location, c_loads = conflict

        resolved = False
        for load in c_loads:
            load.avoid(step, location, c_loads)
            if not list(s.conflicts(c_loads)):
                resolved = True
                break

        if resolved:
            for load in c_loads:
                loads[load.id] = load.path[:]

        s = Schedule(graph, loads)
        loads = {lid: load.path for lid, load in s.loads.items()}
        conflicts = list(s.conflicts())

    print(good, "good,", bad, "bad", flush=True)

    if final_state not in movements:
        raise Exception("No solution found")

    steps, best_path = movements.shortest_path(initial_state, final_state)
    moves = path_to_moves(best_path)
    return moves


# collection of solution methods for the routing problem.
# insert, delete, append or substitute with your own methods as required.
methods = [
    # action_resolve,
    # avoid_resolve,
    # loop_resolve,
    # train_resolve,
    dfs_resolve,
    bfs_resolve
]

