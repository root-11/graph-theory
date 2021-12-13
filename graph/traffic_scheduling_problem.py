from time import process_time
from graph import Graph
from bisect import insort
from itertools import product

__description__ = """
We've decided to refer to the optimisation problem of finding
the fewest number of moves that resolve a traffic jam as
a traffic scheduling problem.
"""


class UnSolvable(ValueError):
    """ Value Error raised if the inputs have no solution. """
    pass


class NoSolution(ValueError):
    """ Value Error raised if the method could not identify a valid solution """
    pass


class Load(object):
    _empty = frozenset()

    __slots__ = ["id", "start", "ends", "prohibited"]

    def __init__(self, id, start, ends=None, prohibited=None):
        """
        :param id: unique load reference
        :param start: start node
        :param ends: end node(s)
            - list, set, frozenset, tuples are interpreted as a number of candidate destinations.
            - all other types are interpreted as 1 destination.
        :param prohibited: iterable with nodes that cannot be in the solution.
        """
        self.id = id
        self.start = start
        if isinstance(ends, frozenset):
            if not ends:
                raise ValueError(f"end is an empty {type(ends)}?")
            self.ends = ends
        elif isinstance(ends, (list, set, tuple)):
            if not ends:
                raise ValueError(f"end is an empty {type(ends)}?")
            self.ends = frozenset(ends)
        elif ends is None:
            self.ends = frozenset([start])
        else:
            self.ends = frozenset([ends])
        assert isinstance(self.ends, frozenset), "bad logic!"

        if prohibited is not None:
            if not isinstance(prohibited, (list, frozenset, set, tuple)):
                raise TypeError(f"Got {type(prohibited)}, expected list, set or tuple")
            self.prohibited = frozenset(prohibited)
        else:
            self.prohibited = Load._empty  # link to an empty frozen set.

    def __str__(self):
        return f"Load(id={self.id}, start={self.start}, end={self.ends}, prohibited={self.prohibited})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Load):
            raise TypeError
        return all([
            self.start == other.start,
            self.ends == other.ends,
            self.prohibited == other.prohibited
        ])


# ----------------- #
# The main solver   #
# ----------------- #
def jam_solver(graph, loads, timeout=None, synchronous_moves=True, return_on_first=False):
    """ an ensemble solver for the routing problem.

    :param graph network available for routing.
    :param loads:

    loads_as_list = [
        {'id': 1, 'start': 1, 'end': 3},  # keyword prohibited is missing. That's okay.
        {'id': 2, 'start': 2, 'end': [3, 4, 5], 'prohibited': [7, 8, 9]},
        {'id': 3, 'start': 3, 'end': [4, 5], 'prohibited': [2]}  # node 2 is gateway to off limits.
    ]

    loads_as_dict = {
        1: (1, 3),  # start, end, None
        2: (2, [3, 4, 5], [7, 8, 9]),  # start, end(s), prohibited
        3: (3, [4, 5], [2])
    }

    :param timeout: None, float or int as timeout in milliseconds.
    :param synchronous_moves: bool, return list of moves as concurrent moves.
    :param return_on_first: bool, if True first solution is returned.
    """
    load_set = check_user_input(graph, loads)

    if isinstance(timeout, (int, float)):
        timer = Timer(timeout)
    elif isinstance(timeout, Timer):
        timer = timeout
    elif timeout is None:
        timer = Timer(None)
    else:
        raise TypeError(f"Expect timeout as int or float, not {type(timeout)}")

    distance, path, movements = float('inf'), None, Graph()
    distance_cache = DistanceCache(graph, load_set.values())
    try:

        for method in methods:  # each search algorithm works on the movements-Graph.
            start = process_time()
            try:
                d, p = method(graph, load_set, timer, distance_cache, movements)
                end = process_time()
                print(method.__name__, "| ", round(d, 4), "moves | time", round(end - start, 4))
            except NoSolution:
                end = process_time()
                print(method.__name__, "| no solution | time", round(end - start, 4))
                continue
            if d < distance:
                distance, path = d, p
                if return_on_first:
                    break

    except TimeoutError:

        if path is None:
            if timeout is None:
                raise NoSolution(f"no solution found.")
            else:
                raise UnSolvable(f"no solution found with timeout = {timeout}")
        else:
            pass  # we return the solution we found so far.

    moves = path_to_moves(path)
    if synchronous_moves:
        return moves_to_synchronous_moves(moves, load_set)
    return moves


class Timer(object):
    def __init__(self, timeout=None):
        """
        :param timeout: int/float in milliseconds.
        """
        if timeout is None:
            timeout = float('inf')
        assert isinstance(timeout, (float, int))
        self.limit = timeout
        self.start = process_time()
        self._counter = 0

    def timeout_check(self):
        if self._counter > 0:
            self._counter -= 1
        else:
            self._counter += 1000
            if process_time() - self.start > (self.limit / 1000):
                raise TimeoutError(f"No solution found in {self.limit} ms")


# helpers
def check_user_input(graph, loads):
    """ checks the user inputs to be valid.

    :param graph network available for routing.
    :param loads: dictionary or list with load id and preferred route. Examples:

    loads_as_list = [
        {'id': 1, 'start': 1, 'end': 3},  # keyword prohibited is missing.
        {'id': 2, 'start': 2, 'end': [3, 4, 5], 'prohibited': [7, 8, 9]},
        {'id': 3, 'start': 3, 'end': [4, 5], 'prohibited': [2]}  # gateway to off limits.
        {'id': 4, 'start': 8}  # load 4 is where it needs to be.
    ]

    loads_as_dict = {
        1: (1, 3),  # start, end, None
        2: (2, [3, 4, 5], [7, 8, 9]),  # start, end(s), prohibited
        3: (3, [4, 5], [2]),
        4: (8, ),
    }

    returns: list of Loads
    """
    if not isinstance(graph, Graph):
        raise TypeError(f"expected graph, not {type(graph)}")

    all_loads = {}
    all_nodes = set()
    if isinstance(loads, list):
        for d in loads:
            if not isinstance(d, dict):
                raise TypeError(f"Got {type(d)}, expected dict.")
            L = Load(**d)
            all_loads[L.id] = L
            all_nodes.add(L.start)
            all_nodes.update(L.ends)
    elif isinstance(loads, dict):
        for i, t in loads.items():
            if not isinstance(t, (tuple, list)):
                raise TypeError(f"Got {type(t)}, expected tuple")
            L = Load(i, *t)
            all_loads[L.id] = L
            all_nodes.add(L.start)
            all_nodes.update(L.ends)
    else:
        raise TypeError("loads not recognised. Please see docstring.")

    if not all_nodes.issubset(set(graph.nodes())):  # then something is wrong. Let's analyze to help the programmer...
        diff = all_nodes.difference(set(graph.nodes()))
        for load in all_loads.values():
            if load.start in diff:
                raise ValueError(f"Load {load.id}'s start ({load.start}) is not in the graph.")
            if load.ends.intersection(diff):
                raise ValueError(f"Load {load.id}'s ends ({load.ends.intersection(diff)}) is/are not in the graph.")
            if load.prohibited.intersection(diff):
                raise ValueError(f"Load {load.id}'s prohibited node(s) ({load.prohibited.intersection(diff)}) is/are not in the graph.")

    for load in all_loads.values():
        if load.prohibited:
            gc = graph.copy()
            for n in load.prohibited:
                gc.del_node(n)
        else:
            gc = graph

        for node in gc.breadth_first_walk(start=load.start):
            if node in load.ends:
                break
        else:
            ends = f"any {tuple(load.ends)}" if len(load.ends)>1 else f"{list(load.ends)[0]}"
            raise UnSolvable(f"load {load.id} has no path from {load.start} to {ends}")
    return all_loads


def path_to_moves(path):
    """
    translate path into a motion sequence.
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


def moves_to_synchronous_moves(moves, loads):
    """ translates the list of moves returned from the traffic jam solver to a list
    of moves that can be made concurrently.

    :param moves: list of loads and moves, e.g. [{1: (2,3)}, {2:(1,2)}, ... ]
    :param loads: dict with loads and paths, e.g. {1: [2,3,4], 2: [1,2,3], ... }
    :return: list of synchronous loads and moves, e.g. [{1:(2,3), {2:(1,2}}, {1:(3,4), 2:(2,3)}, ...]
    """
    moves = [(k,) + v for move in moves for k, v in move.items()]  # create independent copy
    assert isinstance(loads, dict)
    assert all(isinstance(ld, Load) for ld in loads.values())

    occupied_locations = {L.start for L in loads.values()}  # loads are required in case that a load doesn't move.
    synchronous = []

    while moves:
        current_moves = {}
        for move in moves[:]:
            load, n1, n2 = move
            if load in current_moves:
                break
            if n2 in occupied_locations:
                continue
            current_moves[load] = (n1, n2)
            occupied_locations.remove(n1)
            occupied_locations.add(n2)
            moves.remove(move)
        synchronous.append(current_moves)
    return synchronous


# Search methods

def bfs_resolve(graph, loads, timer, distance_cache, movements):
    """
    calculates the solution to the transshipment problem by
    constructing the solution space as a finite state machine
    and then finding the shortest path through the fsm from the
    initial state to the desired state.

    :param graph network available for routing.
    :param loads: dictionary with load id and preferred route. Example:

        loads = {1: [1, 2, 3], 2: [3, 2, 1]}

    :param timer: Instance of Timer

    """
    check_inputs(graph, loads, timer, distance_cache, movements)

    initial_state = tuple(((ld.id, ld.start) for ld in loads.values()))

    if not movements.nodes():
        states = [(0, 0, initial_state)]  # 0,0 is distance_traveled, distance left
    else:
        states = [(0, 0, s) for s in movements.nodes(out_degree=0)]

    solution = None
    while not solution:
        timer.timeout_check()

        state = states.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:

            if solution:
                break
            timer.timeout_check()

            load = loads[load_id]
            options = sorted((d, e) for s, e, d in graph.edges(from_node=location)
                             if e not in occupied and e not in load.prohibited)

            for distance, option in options:
                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)

                if new_state in movements:  # abandon branch
                    continue
                else:
                    movements.add_edge(state, new_state, distance)
                    states.append(new_state)

                check = [loc in loads[lid].ends for lid, loc in new_state]
                if all(check):
                    solution = new_state
                    break

        if not states:
            raise Exception("No solution found")

    return movements.shortest_path(initial_state, solution)


def shortest_path_multiple_ends(movements, start, ends):
    """ helper for bidirectional search on movement graph.
    :param movements:
    :param start:
    :param ends:
    :return:
    """
    assert isinstance(movements, Graph)
    d_min, p_min = float('inf'), None
    for end in ends:
        d, p = movements.shortest_path(start, end)
        if d < d_min:  # then this solution is better than the previous.
            d_min = d
            p_min = p
    return d_min, p_min


def possible_end_state_gen(loads):
    """ helper for bidirectional search.
    generates end state for loads with multiple destinations
    """
    destinations = [list(ld.ends) for ld in loads.values()]
    for combo in product(*destinations):
        if len(combo) != len(destinations):
            continue  # it's a duplicate.
        yield combo


class LoadPath(object):
    def __init__(self, graph, load):
        self.graph = graph
        self.load = load
        self.current_location = load.start
        paths = [graph.shortest_path(load.start, end) for end in load.ends]
        paths.sort()  # shortest on top.
        d,p = paths[0]
        self.path = p

    def to_load(self):
        if self.current_location in self.load.ends:
            ends = {self.current_location}
        else:
            ends = self.load.ends
        return Load(self.id, start=self.current_location, ends=ends, prohibited=self.load.prohibited)

    @property
    def start(self):
        return self.load.start

    @property
    def id(self):
        return self.load.id

    def where_to(self):
        ix = self.path.index(self.current_location)
        try:
            return self.path[ix + 1]
        except IndexError:
            return self.current_location

    def at_destination(self):
        return self.current_location in self.load.ends

    def find_alternative_route(self, obstacles):
        paths = [self.graph.shortest_path(self.current_location, end, avoids=obstacles) for end in self.load.ends]
        paths.sort()  # shortest on top.
        d, p = paths[0]
        if p:
            self.path = p
        else:  # there's no path. Perhaps I need to step out of the way?
            # options = set(e for s, e, d in self.graph.edges(from_node=self.current_location) if e not in obstacles and e not in self.load.prohibited)
            # if options and options.intersection(set(self.path)):
            #     choice = options.intersection(set(self.path)).pop()
            #     ix = self.path.index(self.current_location)
            #     self.path.insert(ix+1, choice)
            # else:
            pass  # wait.




def bi_directional_progressive_bfs(movements, graph, loads, timer=None):
    """ Bi-directional search which searches to the end of open options for each load.

    :param graph network available for routing.
    :param loads: dictionary with loads
    :param timer: Instance of Timer
    """
    initial_state = tuple(((ld.id, ld.start) for ld in loads.values()))
    final_state = tuple(((ld.id, ld.ends) for ld in loads.values()))

    forward_queue = [initial_state]
    forward_states = {initial_state}
    reverse_queue = [final_state]
    reverse_states = {final_state}

    solved = False
    while not solved:
        if timer is not None:
            timer.timeout_check()

        # forward
        if not forward_queue:
            raise NoSolution
        state = forward_queue.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:
            if solved:
                break
            options = {e: state for s, e, d in graph.edges(from_node=location) if e not in occupied}
            if not options:
                continue

            visited = {i for i in occupied}
            while options:
                option = list(options.keys())[0]
                old_state = options.pop(option)  # e from s,e,d

                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in old_state)
                if new_state not in movements:
                    forward_queue.append(new_state)

                movements.add_edge(old_state, new_state, 1)
                forward_states.add(new_state)

                visited.add(option)
                options.update({e: new_state for s, e, d in graph.edges(from_node=option) if e not in visited})

                if new_state in reverse_states:
                    solved = True
                    break

        # backwards
        if not reverse_queue:
            raise NoSolution
        state = reverse_queue.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:
            if solved:
                break

            options = {s: state for s, e, d in graph.edges(to_node=location) if s not in occupied}
            if not options:
                continue

            visited = {i for i in occupied}
            while options:
                option = list(options.keys())[0]  # s from s,e,d
                old_state = options.pop(option)

                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in old_state)

                if new_state not in movements:  # add to queue
                    reverse_queue.append(new_state)

                movements.add_edge(new_state, old_state, 1)
                reverse_states.add(new_state)

                visited.add(option)
                options.update({s: new_state for s, e, d in graph.edges(to_node=option) if s not in visited})

                if new_state in forward_states:
                    solved = True
                    break

    return movements.shortest_path(initial_state, final_state)


def bi_directional_bfs(movements, graph, loads, timer=None):
    """ calculates the solution to the transshipment problem using BFS
    from both initial and final state

    :param graph network available for routing.
    :param loads: dictionary with load id and preferred route. Example:
    :param timer: Instance of Timer
    """
    if not isinstance(movements, Graph):
        raise TypeError
    if not isinstance(graph, Graph):
        raise TypeError
    if not isinstance(loads, dict):
        raise TypeError
    if not all(isinstance(i, Load) for i in loads.values()):
        raise TypeError
    if not isinstance(timer, (type(None), Timer)):
        raise TypeError

    initial_state = tuple(((ld.id, ld.start) for ld in loads.values()))

    final_state = []
    for load in loads.values():
        nn = [graph.shortest_path(load.start, end, avoids=load.prohibited) for end in load.ends]
        nn.sort()
        d, p = nn[0]
        final_state.append((load.id, p[-1]))
    final_state = tuple(final_state)

    movements = Graph()
    forward_queue = [initial_state]
    forward_states = {initial_state}
    reverse_queue = [final_state]
    reverse_states = {final_state}

    solution = None

    while solution is None:
        if timer is not None:
            timer.timeout_check()

        # forward
        if not forward_queue:
            raise NoSolution("No solution found")
        state = forward_queue.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:
            if solution:
                break
            load = loads[load_id]
            options = sorted((d, e) for s, e, d in graph.edges(from_node=location) if e not in occupied and e not in load.prohibited)

            for distance, option in options:
                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)
                if new_state not in movements:
                    forward_queue.append(new_state)

                movements.add_edge(state, new_state, distance)
                forward_states.add(new_state)

                if new_state in reverse_states:
                    solution = new_state
                    break

        # backwards
        if not reverse_queue:
            raise NoSolution("No solution found")
        state = reverse_queue.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:
            load = loads[load_id]
            if solution:
                break
            options = sorted((d, e) for s, e, d in graph.edges(from_node=location) if s not in occupied and e not in load.prohibited)
            for distance, option in options:
                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)

                if new_state not in movements:  # add to queue
                    reverse_queue.append(new_state)

                movements.add_edge(new_state, state, distance)
                reverse_states.add(new_state)

                if new_state in forward_states:
                    solution = True
                    break

    return movements.shortest_path(initial_state, final_state)


class DistanceCache(object):
    """ Helper for the hill climb so that number of distance lookups are minimised."""
    def __init__(self, graph, loads):
        if not isinstance(graph, Graph):
            raise TypeError
        self.graph = graph
        self.cache = {}
        self.loads = {load.id: load for load in loads}

    def state(self, state):
        d_min = self.cache.get(state, None)
        if d_min is None:
            d_min = sum(self.load(load_id, loc) for load_id, loc in state)
            self.cache[state] = d_min
        return d_min

    def load(self, load_id, loc):
        key = (load_id, loc)
        d_min = self.cache.get(key, None)
        if d_min is None:
            load = self.loads[load_id]
            d_min = min([d for d, p in (self.graph.shortest_path(loc, e, avoids=load.prohibited) for e in load.ends)])
            self.cache[key] = d_min
        return d_min


def check_inputs(graph, loads,timer,distance_cache, movements):
    if not isinstance(graph, Graph):
        raise TypeError
    if not isinstance(loads, dict):
        raise TypeError
    if not all(isinstance(i, Load) for i in loads.values()):
        raise TypeError
    if not isinstance(timer, Timer):
        raise TypeError
    if not isinstance(distance_cache, DistanceCache):
        raise TypeError
    if not isinstance(movements, Graph):
        raise TypeError


def hill_climb(graph, loads, timer, distance_cache, movements):
    """ A purist hill-climbing algorithm
    :param graph: graph network available for routing.
    :param loads: dict with Loads

    :param timer: Instance of Timer
    :return: list of moves.
    """
    check_inputs(graph, loads, timer, distance_cache, movements)

    initial_state = tuple(((ld.id, ld.start) for ld in loads.values()))

    states = [(0, initial_state)]
    best_score = float('inf')
    solution = None
    while not solution:
        timer.timeout_check()

        score, state = states.pop(0)
        occupied = {i[1] for i in state}

        for load_id, location in state:
            if solution:
                break

            load = loads[load_id]

            options = sorted((d, e) for s, e, d in graph.edges(from_node=location) if e not in occupied and e not in load.prohibited)

            for distance, option in options:

                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)
                timer.timeout_check()
                if new_state in movements:  # abandon branch, it has already been checked.
                    continue
                else:
                    movements.add_edge(state, new_state, distance)

                check = [loc in loads[lid].ends for lid, loc in new_state]
                if all(check):
                    solution = new_state
                    break

                new_score = distance_cache.state(new_state)
                if new_score > best_score:
                    continue
                best_score = new_score
                insort(states, (new_score, new_state))
            states = [(score, s) for score, s in states if score <= best_score]

        if not states:
            raise NoSolution("No solution found")  # hill climbing doesn't lead to a solution

    return movements.shortest_path(initial_state, solution)


def simple_path(graph, loads, timer, distance_cache, movements):
    """A very greed algorithm.

    :param graph:
    :param loads:
    :param timer:
    :param distance_cache:
    :param movements:
    :return:
    """
    check_inputs(graph, loads, timer, distance_cache, movements)

    all_loads = [LoadPath(graph, load) for load in loads.values()]
    occupied = {ld.start for ld in loads.values()}

    start = [(load.id, load.current_location) for load in all_loads]

    while True:
        # ask each load where they would like to go.
        end = []
        distance = 0
        for load in all_loads:

            timer.timeout_check()

            if load.at_destination():
                next_location = load.current_location
            else:
                next_location = load.where_to()
                if next_location in occupied:  # load will have to find another route or wait.

                    old_route = load.path[:]

                    load.find_alternative_route(obstacles={o for o in occupied if o != load.current_location})
                    if load.path == old_route or load.where_to() in occupied:  # then it'll have to wait.
                        next_location = load.current_location
                    else:
                        next_location = load.where_to()  # then it moves somewhere.

                distance += graph.edge(load.current_location, next_location, 0)

                occupied.remove(load.current_location)
                occupied.add(next_location)
                load.current_location = next_location

            end.append((load.id, next_location))

        a, b = tuple(start), tuple(end)
        if movements.edge(a, b) is None:  # then we're progressing.
            movements.add_edge(a, b, distance)
        else:  # we are not progressing!
            break
        start = end[:]

    if all(load.at_destination() for load in all_loads):
        initial = tuple((load.id, load.start) for load in all_loads)
        final = tuple((load.id, load.current_location) for load in all_loads)
        return movements.shortest_path(initial, final)

    else:
        raise NoSolution("No solution found")  # hill climbing doesn't lead to a solution


def bidirectional_breadth_first_search(graph, loads, timer, distance_cache, movements):
    """
    :param graph:
    :param loads:
    :param timer:
    :param distance_cache:
    :param movements:
    :return:
    """
    check_inputs(graph, loads, timer, distance_cache, movements)

    INF = float('inf')
    forward_max_distance = float('inf')
    reverse_max_distance = float('inf')

    initial_state = tuple(((ld.id, ld.start) for ld in loads.values()))
    forward_queue = [(0, initial_state)]
    forward_edge = {initial_state}

    reverse_queue = []
    reverse_edge = set()
    final_states = set()

    for combo in possible_end_state_gen(loads):
        t = tuple((ld, loc) for ld, loc in zip(loads, combo))
        final_states.add(t)
        reverse_edge.add(t)
        reverse_queue.append((0, t))

    while forward_queue or reverse_queue:
        timer.timeout_check()

        if forward_queue:  # forward ....
            distance_traveled, state = forward_queue.pop(0)

            occupied = {i[1] for i in state}
            for load_id, location in state:
                load = loads[load_id]

                options = sorted((d, e) for s, e, d in graph.edges(from_node=location) if
                                 e not in occupied and e not in load.prohibited)

                for distance, option in options:
                    new_distance_traveled = distance_traveled + distance

                    new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)
                    # if new_state in forward_edge:
                    #     continue
                    if distance < movements.edge(state, new_state, INF):
                        movements.add_edge(state, new_state, distance)
                        forward_edge.add(new_state)
                        insort(forward_queue, (new_distance_traveled, new_state))
                    else:
                        continue

                    if new_state in reverse_edge:
                        fmd, _ = movements.shortest_path(initial_state, new_state)
                        forward_max_distance = min(fmd, forward_max_distance)
                        spme, _ = shortest_path_multiple_ends(movements, new_state, final_states)
                        reverse_max_distance = min(spme, reverse_max_distance)

                        forward_queue = [(d, s) for d, s in forward_queue if d <= forward_max_distance]
                        forward_queue = [(d, s) for d, s in forward_queue if d <= forward_max_distance]

        if reverse_queue:  # backward...
            distance_traveled, state = reverse_queue.pop(0)
            occupied = {i[1] for i in state}
            for load_id, location in state:
                load = loads[load_id]

                options = sorted((d, s) for s, e, d in graph.edges(to_node=location) if
                                 s not in occupied and s not in load.prohibited)

                for distance, option in options:
                    new_distance_traveled = distance_traveled + distance

                    new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)
                    # if new_state in reverse_edge:
                    #     continue
                    if distance < movements.edge(new_state, state, INF):
                        movements.add_edge(new_state, state, distance)
                        reverse_edge.add(new_state)
                        insort(reverse_queue, (new_distance_traveled, new_state))
                    else:
                        continue

                    if new_state in forward_edge:
                        fmd, _ = movements.shortest_path(initial_state, new_state)
                        forward_max_distance = min(fmd, forward_max_distance)
                        spme, _ = shortest_path_multiple_ends(movements, new_state, final_states)
                        reverse_max_distance = min(spme, reverse_max_distance)

                        reverse_queue = [(d, s) for d, s in reverse_queue if d <= reverse_max_distance]

    min_distance, min_distance_path = shortest_path_multiple_ends(movements, initial_state, final_states)
    return min_distance, min_distance_path


def breadth_first_search(graph, loads, timer, distance_cache, movements):
    """
    :param graph:
    :param loads:
    :param timer:
    :param distance_cache:
    :param movements:
    :return:
    """
    check_inputs(graph, loads, timer, distance_cache, movements)

    initial_state = tuple(((ld.id, ld.start) for ld in loads.values()))

    if not movements.nodes():
        states = [(0, 0, initial_state)]  # 0,0 is distance_traveled, distance left
    else:
        states = [(0, 0, s) for s in movements.nodes(out_degree=0)]

    min_distance, min_distance_path = float('inf'), None

    while states:
        timer.timeout_check()

        # get the shortest distance traveled up first
        distance_traveled, distance_left, state = states.pop(0)

        occupied = {i[1] for i in state}
        for load_id, location in state:
            load = loads[load_id]
            # if location in load.ends:
            #     continue

            options = sorted(
                (d, e) for s, e, d in graph.edges(from_node=location) if e not in occupied and e not in load.prohibited)

            for distance, option in options:
                timer.timeout_check()

                new_distance_traveled = distance_traveled + distance

                if new_distance_traveled > min_distance:  # the solution will be worse than best-known solution.
                    continue

                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)
                if movements.edge(state, new_state, float('inf')) < distance:  # the option has already been explored.
                    continue
                movements.add_edge(state, new_state, distance)

                new_distance_left = distance_cache.state(new_state)
                if (new_distance_traveled, new_distance_left, new_state) in states:  # the edge is already known.
                    continue

                insort(states, (new_distance_traveled, new_distance_left, new_state))

                check = [loc in loads[lid].ends for lid, loc in new_state]
                if all(check):  # then all loads are in a valid final state.
                    d, p = movements.shortest_path(initial_state, new_state)
                    if d < min_distance:  # then this solution is better than the previous.
                        min_distance, min_distance_path = d, p
                        # finally purge min distance.
                        states = [(a, b, c) for a, b, c in states if a < min_distance]

    if not min_distance_path:
        raise NoSolution

    return min_distance, min_distance_path


# collection of solution methods for the routing problem.
# insert, delete, append or substitute with your own methods as required.
methods = [
    hill_climb,  # cheap check.
    simple_path,
    # bi_directional_progressive_bfs,  # <-- the fastest, but not always the best method.
    # bi_directional_bfs,  # <-- best method so far.
    bidirectional_breadth_first_search,
    breadth_first_search,
    # bfs_resolve,  # very slow, but will eventually find the best solution.
]



