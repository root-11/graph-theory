from time import process_time
from graph import Graph


__description__ = """
We've decided to refer to the optimisation problem of finding
the fewest number of moves that resolve a traffic jam as
a traffic scheduling problem.
"""


class Load(object):
    _empty = frozenset()

    __slots__ = ["id", "start", "ends", "prohibited"]

    def __init__(self, id, start, ends, prohibited=None):
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
def jam_solver(graph, loads, timeout=None):
    """ an ensemble solver for the routing problem.

    :param graph network available for routing.
    :param loads:

    loads_as_list = [
        {'id': 1, 'start': 1, 'end': 3},  # keyword prohibited is missing.
        {'id': 2, 'start': 2, 'end': [3, 4, 5], 'prohibited': [7, 8, 9]},
        {'id': 3, 'start': 3, 'end': [4, 5], 'prohibited': [2]}  # gateway to off limits.
    ]

    loads_as_dict = {
        1: (1, 3),  # start, end, None
        2: (2, [3, 4, 5], [7, 8, 9]),  # start, end(s), prohibited
        3: (3, [4, 5], [2])
    }

    :param timeout: None, float or int timeout in milliseconds.
    """
    check_user_input(graph, loads)
    if timeout:
        assert isinstance(timeout, (int, float))

    moves = None
    for method in methods:
        try:
            moves = method(graph, loads, timeout)
        except (TimeoutError, Exception) as e:
            if isinstance(e, Exception):
                assert str(e) == "No solution found", f"{e} instead of No solution found"
            continue
        if moves:
            return moves
    return moves


class Timer(object):
    def __init__(self, timeout=None):
        if timeout is None:
            timeout = float('inf')
        assert isinstance(timeout, (float, int))
        self.limit = timeout
        self.start = process_time()

    def timeout_check(self):
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
    ]

    loads_as_dict = {
        1: (1, 3),  # start, end, None
        2: (2, [3, 4, 5], [7, 8, 9]),  # start, end(s), prohibited
        3: (3, [4, 5], [2])
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

    if not all_nodes.issubset(set(graph.nodes())):
        raise ValueError("Some load nodes are not in the graph.")

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
            raise ValueError(f"load {load.id} has no path from {load.start} to {load.ends}")
    return all_loads


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


def bfs_resolve(graph, loads, timeout=None):
    """
    calculates the solution to the transshipment problem by
    constructing the solution space as a finite state machine
    and then finding the shortest path through the fsm from the
    initial state to the desired state.

    :param graph network available for routing.
    :param loads: dictionary with load id and preferred route. Example:

        loads = {1: [1, 2, 3], 2: [3, 2, 1]}

    :param timeout: None, float or int timeout in milliseconds.

    """
    initial_state = tuple(((load_id, route[0]) for load_id, route in loads.items()))
    final_state = tuple(((load_id, route[-1]) for load_id, route in loads.items()))
    movements = Graph()

    states = [initial_state]

    timer = Timer(timeout=timeout)
    solved = False

    while not solved:
        timer.timeout_check()

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


def bi_directional_progressive_bfs(graph, loads, timeout=None):
    """ Bi-directional search which searches to the end of open options for each load.

    :param graph network available for routing.
    :param loads: dictionary with loads
    :param timeout: None, float or int timeout in milliseconds.
    """
    initial_state = tuple(((ld.id, ld.start) for ld in loads.values()))
    final_state = tuple(((ld.id, ld.ends) for ld in loads.values()))

    movements = Graph()
    forward_queue = [initial_state]
    forward_states = {initial_state}
    reverse_queue = [final_state]
    reverse_states = {final_state}

    timer = Timer(timeout=timeout)
    solved = False

    while not solved:
        timer.timeout_check()

        # forward
        if not forward_queue:
            raise Exception("No solution found")
        state = forward_queue.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:
            if solved:
                break
            options = {e: state for s, e, d in graph.edges(from_node=location) if e not in occupied}
            if not options:
                continue

            been = {i for i in occupied}
            while options:
                option = list(options.keys())[0]
                old_state = options.pop(option)  # e from s,e,d

                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in old_state)
                if new_state not in movements:
                    forward_queue.append(new_state)

                movements.add_edge(old_state, new_state, 1)
                forward_states.add(new_state)

                been.add(option)
                options.update({e: new_state for s, e, d in graph.edges(from_node=option) if e not in been})

                if new_state in reverse_states:
                    solved = True
                    break

        # backwards
        if not reverse_queue:
            raise Exception("No solution found")
        state = reverse_queue.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:
            if solved:
                break

            options = {s: state for s, e, d in graph.edges(to_node=location) if s not in occupied}
            if not options:
                continue

            been = {i for i in occupied}
            while options:
                option = list(options.keys())[0]  # s from s,e,d
                old_state = options.pop(option)

                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in old_state)

                if new_state not in movements:  # add to queue
                    reverse_queue.append(new_state)

                movements.add_edge(new_state, old_state, 1)
                reverse_states.add(new_state)

                been.add(option)
                options.update({s: new_state for s, e, d in graph.edges(to_node=option) if s not in been})

                if new_state in forward_states:
                    solved = True
                    break

    steps, best_path = movements.shortest_path(initial_state, final_state)
    moves = path_to_moves(best_path)
    return moves


def bi_directional_bfs(graph, loads, timeout=None):
    """ calculates the solution to the transshipment problem using BFS
    from both initial and final state

    :param graph network available for routing.
    :param loads: dictionary with load id and preferred route. Example:

        loads = {1: [1, 2, 3], 2: [3, 2, 1]}

    :param timeout: None, float or int timeout in milliseconds.
    """
    initial_state = tuple(((load_id, route[0]) for load_id, route in loads.items()))
    final_state = tuple(((load_id, route[-1]) for load_id, route in loads.items()))

    movements = Graph()
    forward_queue = [initial_state]
    forward_states = {initial_state}
    reverse_queue = [final_state]
    reverse_states = {final_state}

    timer = Timer(timeout=timeout)
    solved = False

    while not solved:
        timer.timeout_check()

        # forward
        if not forward_queue:
            raise Exception("No solution found")
        state = forward_queue.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:
            if solved:
                break
            options = [e for s, e, d in graph.edges(from_node=location) if e not in occupied]
            for option in options:
                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)
                if new_state not in movements:
                    forward_queue.append(new_state)

                movements.add_edge(state, new_state, 1)
                forward_states.add(new_state)

                if new_state in reverse_states:
                    solved = True
                    break

        # backwards
        if not reverse_queue:
            raise Exception("No solution found")
        state = reverse_queue.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:
            if solved:
                break
            options = [s for s, e, d in graph.edges(to_node=location) if s not in occupied]
            for option in options:
                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)

                if new_state not in movements:  # add to queue
                    reverse_queue.append(new_state)

                movements.add_edge(new_state, state, 1)
                reverse_states.add(new_state)

                if new_state in forward_states:
                    solved = True
                    break

    steps, best_path = movements.shortest_path(initial_state, final_state)
    moves = path_to_moves(best_path)
    return moves


def pure_hill_climbing_algorithm(graph, loads, timeout=None):
    """ A purist hill-climbing algorithm
    :param graph: graph network available for routing.
    :param loads: dict with Loads

    :param timeout: None, float or int timeout in milliseconds.
    :return: list of moves.
    """
    if not isinstance(graph, Graph):
        raise TypeError
    if not isinstance(loads, dict):
        raise TypeError
    if not all(isinstance(i, Load) for i in loads.values()):
        raise TypeError

    initial_state = tuple(((ld.id, ld.start) for ld in loads.values()))
    movements = Graph()

    states = [(0, initial_state)]

    timer = Timer(timeout=timeout)

    solution = None
    while not solution:
        timer.timeout_check()

        score, state = states.pop(0)
        occupied = {i[1] for i in state}
        for load_id, location in state:
            load = loads[load_id]
            if solution: break

            options = (e for s, e, d in graph.edges(from_node=location) if
                       e not in occupied and
                       e not in load.prohibited)
            for option in options:
                new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)

                if new_state in movements:  # abandon branch
                    continue

                movements.add_edge(state, new_state, 1)

                check = [loc in loads[lid].ends for lid, loc in new_state]
                if all(check):
                    solution = new_state
                    break

                new_score = sum([1 for c in check if c])
                if new_score < score:
                    continue
                states.append((new_score, new_state))
                states = [(score, s) for score, s in states if score >= new_score]

        if not states:
            raise Exception("No solution found")  # hill climbing doesn't lead to a solution

    steps, best_path = movements.shortest_path(initial_state, solution)
    moves = path_to_moves(best_path)
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
    assert all(isinstance(t, (list, tuple)) for t in loads.values())

    occupied_locations = {L[0] for L in loads.values()}  # loads are required in case that a load doesn't move.
    synchronuous_moves = []

    while moves:
        current_moves = {}
        for move in moves[:]:
            load, n1, n2 = move
            if load in current_moves:
                break
            if n2 in occupied_locations:
                continue
            current_moves[load] = (n1,n2)
            occupied_locations.remove(n1)
            occupied_locations.add(n2)
            moves.remove(move)
        synchronuous_moves.append(current_moves)
    return synchronuous_moves


# collection of solution methods for the routing problem.
# insert, delete, append or substitute with your own methods as required.
methods = [
    pure_hill_climbing_algorithm,  # cheap check.
    bi_directional_progressive_bfs,  # <-- the fastest, but not always the best method.
    bi_directional_bfs,  # <-- best method so far.
    bfs_resolve,  # very slow, but will eventually find the best solution.
]



