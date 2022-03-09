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


class StopCondition(Exception):
    """ exception used to stop the search """
    pass


class Timer(object):
    def __init__(self, timeout=None):
        """
        :param timeout: int/float in milliseconds.
        """
        if timeout is None:
            timeout = float('inf')
        if not isinstance(timeout, (float, int)):
            raise ValueError(f"timeout is {type(timeout)} not int or float > 0")
        if timeout < 0:
            raise ValueError(f"timeout must be >0, but was {timeout}")

        self.limit = timeout
        self.start = process_time()
        self._counter = 0
        self._expired = False

    def expired(self):
        """ returns bool"""
        if self._expired:
            return True

        # we use a counter as there is no reason to check the time 100_000 times/second.
        if self._counter > 0:
            self._counter -= 1
            return False
        # The counter was zero, so now we check the time and reset the counter.
        self._counter = 100
        if process_time() - self.start > (self.limit / 1000):
            self._expired = True
        return False


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

    assignment_options = {load.id: frozenset(load.ends) for load in all_loads.values() }
    if not is_ap_solvable(assignment_options):
        raise UnSolvable(f"There are not enough ends for all the loads to be assigned to a destination.")

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


def is_ap_solvable(assignments):
    """
    A number of loads need to be assigned to a destination.
    The loads have preferences, f.x.

        A = {1,2}, B = {2,3}, C = {1,2,3}  # solveable
        A = {1,2}, B = {1,3}, C = {1}      # solveable
        A = {1,2}, B = {1,2}, C = {1}      # not solveable.

    This method checks if the assignment is possible
    """
    if not isinstance(assignments, dict):
        raise TypeError
    if not all(isinstance(i, (frozenset, set)) for i in assignments.values()):
        raise TypeError

    all_ends = set().union(*assignments.values())

    assignment = {}

    for load_id, ends in sorted(assignments.items(), key=lambda x: len(x[-1])):
        options = set(ends).intersection(all_ends)
        if not options:
            return False
        selection = options.pop()
        all_ends.remove(selection)
        assignment[load_id] = selection
    return True


def path_to_moves(path):
    """
    translate path with states into a motion sequence.
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


class State(object):
    def __init__(self, loads, distance=0, gradient=float('inf')):
        self.loads = loads
        self._hash = hash(self.loads)  # by pre-calculating the hash, we save cpu time later.
        self.distance = distance
        self.gradient = gradient
        self._d = None  # for most cases this isn't needed.

    def __str__(self):
        return f"State({self.loads}, {self.distance})"

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        if self.gradient == other.gradient:
            return self.distance < other.distance
        else:
            return self.gradient < other.gradient

    def __iter__(self):
        for lid, loc in self.loads:
            yield lid, loc

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        # this is actually required for dict's to work. For more see example on:
        # https://stackoverflow.com/questions/9010222/why-can-a-python-dict-have-multiple-keys-with-the-same-hash?noredirect=1&lq=1
        return self._hash == other._hash

    def occupied(self):
        return {i[1] for i in self.loads}


class JamSolver(object):
    def __init__(self, graph, loads, timer):
        if not isinstance(graph, Graph):
            raise TypeError
        if not isinstance(loads, dict):
            raise TypeError
        if not all(isinstance(i, Load) for i in loads.values()):
            raise TypeError
        if not all(lid == load.id for lid, load in loads.items()):
            raise ValueError
        if not isinstance(timer, Timer):
            raise TypeError

        self.graph = graph
        self.movements = Graph()
        self.loads = loads
        self.timer = timer

        initial_state = State(loads=tuple((ld.id, ld.start) for ld in self.loads.values()), distance=0)
        self.start = initial_state
        self.ends = set()

        self.movements.add_node(initial_state)

        self.forward_queue = [initial_state]                # this is the working queue with priority.
        self.forward_edge = {initial_state: initial_state}  # this is a duplicate of items in the work queue
        self.forward_visited = set()                        # we don't want to spend CPU time on this.

        self.reverse_queue = []
        self.reverse_edge = {}
        self.reverse_visited = set()

        self.max_search_distance = float('inf')

        self.final_states = set()

        for state in self._end_state_gen():
            self.ends.add(state)
            self.reverse_queue.append(state)
            self.reverse_edge[state] = state
            self.final_states.add(state)

        self.distance_maps = {}
        # The distance maps are used as proxy for simulated annealing.
        # The closer a load is to any of it's destinations, the lower the temperature.
        # when all loads have reach a destination, the temperature is zero.
        for load_id, load in self.loads.items():
            self.distance_maps[load_id] = self.graph.distance_map(ends=load.ends, reverse=True)

        self.done = False
        self.return_on_first = False

    def __str__(self):
        s = "timed out" if self.timer.expired() else "running"
        return f"<{self.__class__.__name__}> {s}"

    def _distance(self, load_id, location):
        try:
            return self.distance_maps[load_id][location]
        except KeyError:
            return max(self.distance_maps[load_id].values()) + 1

    def solve(self, return_on_first=False):
        if not isinstance(return_on_first, bool):
            raise TypeError
        self.return_on_first = return_on_first

        try:
            self._search()
        except StopCondition as e:
            print(str(e))
        return self._shortest_path_multiple_ends()

    def _end_state_gen(self):
        ids, destinations = [], []
        for load in self.loads.values():
            ids.append(load.id)
            destinations.append(list(load.ends))

        for combo in product(*destinations):
            if len(set(combo)) != len(destinations):
                continue  # it's a duplicate.
            state = tuple((lid, loc) for lid, loc in zip(ids, combo))
            yield State(state)

    def _search(self):
        while not self.timer.expired():

            self._find_forward_options()  # forward search
            self._find_reverse_options()  # reverse search

            if not any((self.forward_queue, self.reverse_queue)):
                raise StopCondition("queue exhausted")
                # note: This may mean the solution-landscape is exhausted. Not that something is wrong.

    def _find_forward_options(self):
        if not self.forward_queue:
            return
        state = self.forward_queue.pop(0)
        self.forward_edge.pop(state)
        self.forward_visited.add(state)

        occupied = state.occupied()
        for load_id, location in state:
            load = self.loads[load_id]

            options = sorted((d, e) for s, e, d in self.graph.edges(from_node=location)
                             if e not in occupied and e not in load.prohibited)

            for distance, option in options:
                new_distance_traveled = state.distance + distance

                loads = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)
                gradient = sum((self._distance(lid, loc) for lid, loc in loads))
                new_state = State(loads, new_distance_traveled, gradient)

                if self.movements.edge(state, new_state, float('inf')) < distance:
                    continue
                else:
                    self.movements.add_edge(state, new_state, distance)

                if new_state in self.forward_edge:
                    existing_new_state = self.forward_edge.get(new_state)
                    if new_state < existing_new_state:  # replace!
                        self.forward_edge[new_state] = new_state
                        self.forward_queue.remove(existing_new_state)
                        insort(self.forward_queue, new_state)
                    else:
                        pass
                    continue  # case is already queued for testing.
                elif new_state in self.forward_visited or new_state in self.reverse_visited:
                    continue  # seen before
                else:
                    self.forward_edge[new_state] = new_state
                    insort(self.forward_queue, new_state)

                if new_state in self.reverse_visited:
                    self._match()

    def _shortest_path_multiple_ends(self):
        """ helper that identifies the fewest number of moves required to reach any
        of the valid end states """
        d_min, p_min = float('inf'), None
        for end in self.final_states:
            if end in self.movements:
                d, p = self.movements.shortest_path(self.start, end)
                if d < d_min:  # then this solution is better than the previous.
                    d_min = d
                    p_min = p
        return d_min, p_min

    def _match(self):
        """ helper that updates the search queues when one finds a matching state in the other."""
        d, p = self._shortest_path_multiple_ends()
        self.max_search_distance = min(d, self.max_search_distance)

        self.forward_visited.update({s for s in self.forward_queue if s.distance > self.max_search_distance})
        self.forward_queue = [s for s in self.forward_queue if s.distance <= self.max_search_distance]
        self.forward_edge = {s: s for s in self.forward_queue}

        self.reverse_visited.update({s for s in self.reverse_queue if s.distance > self.max_search_distance})
        self.reverse_queue = [s for s in self.reverse_queue if s.distance <= self.max_search_distance]
        self.reverse_edge = {s: s for s in self.reverse_queue}

        if self.return_on_first:
            raise StopCondition("solution found")

    def _find_reverse_options(self):
        if not self.reverse_queue:  # backward...
            return
        state = self.reverse_queue.pop(0)
        self.reverse_edge.pop(state)
        self.reverse_visited.add(state)

        occupied = state.occupied()
        for load_id, location in state:
            load = self.loads[load_id]

            options = sorted((d, s) for s, e, d in self.graph.edges(to_node=location) if
                             s not in occupied and s not in load.prohibited)

            for distance, option in options:
                new_distance_traveled = state.distance + distance

                loads = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)
                gradient = sum((self._distance(lid, loc) for lid, loc in loads))
                new_state = State(loads, new_distance_traveled, gradient)

                if self.movements.edge(new_state, state, float('inf')) < distance:
                    continue
                else:
                    self.movements.add_edge(new_state, state, distance)

                if new_state in self.reverse_edge:
                    existing_new_state = self.reverse_edge.get(new_state)
                    if new_state < existing_new_state:  # replace!
                        self.reverse_edge[new_state] = new_state
                        self.reverse_queue.remove(existing_new_state)
                        insort(self.reverse_queue, new_state)
                    else:
                        pass
                    continue  # case is already queued for testing.
                elif new_state in self.reverse_visited or new_state in self.forward_visited:
                    continue  # seen before
                else:
                    self.reverse_edge[new_state] = new_state
                    insort(self.reverse_queue, new_state)

                if new_state in self.forward_visited:
                    self._match()


def jam_solver(graph, loads, timeout=None, synchronous_moves=True, return_on_first=False):
    """
    The traffic jam solver
    - a bidirectional search algorithm that uses simulated annealing
    to induce bias to accelerate the solvers direction of search.

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

    :param timeout: None or Number of milliseconds.
    :param synchronous_moves: boolean, set to True to return concurrent moves
    :param return_on_first: boolean, tell solver to stop at first valid solution,
                            disregarding whether the solution is the most energy
                            efficient.
    :return: list of dictionaries as sequence of moves. Example:

    solution = [{'b': (5, 6), 'c': (4, 5), 'e': (1, 4)},              # 1st moves.
                {'b': (6, 3), 'c': (5, 6), 'e': (4, 5), 'a': (2, 1)}, # 2nd moves.
                {'b': (3, 2), 'c': (6, 3), 'e': (5, 6)},              # 3rd moves.
                {'e': (6, 9)}]                                        # 4th move.

    """
    all_loads = check_user_input(graph, loads)
    timer = Timer(timeout)
    c = JamSolver(graph, all_loads, timer)
    d, p = c.solve(return_on_first)

    if not p:
        if timer.expired():
            raise UnSolvable(f"no solution found with timeout = {timeout} msecs")
        else:
            raise NoSolution(f"no solution found.")

    moves = path_to_moves(p)
    if synchronous_moves:
        return moves_to_synchronous_moves(moves, all_loads)
    return moves

