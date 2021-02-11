from time import process_time
from graph import Graph


__description__ = """
We've decided to refer to the optimisation problem of finding
the fewest number of moves that resolve a traffic jam as
a traffic scheduling problem.
"""


# ----------------- #
# The main solver   #
# ----------------- #
def jam_solver(graph, loads, timeout=None):
    """ an ensemble solver for the routing problem.

    :param graph network available for routing.
    :param loads: dictionary with load id and preferred route. Example:

        loads = {1: [1, 2, 3], 2: [3, 2, 1]}

    :param timeout: None, float or int timeout in milliseconds.
    """
    check_user_input(graph, loads)
    if timeout:
        assert isinstance(timeout, (int, float))

    moves = None
    for method in methods:
        try:
            moves = method(graph, loads, timeout)
        except TimeoutError:
            pass
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
    :param loads: dictionary with load id and preferred route. Example:

        loads = {1: [1, 2, 3], 2: [3, 2, 1]}
    """
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
    :param loads: dictionary with load id and preferred route. Example:

        loads = {1: [1, 2, 3], 2: [3, 2, 1]}

    :param timeout: None, float or int timeout in milliseconds.
    :return: list of moves.
    """
    initial_state = tuple(((load_id, route[0]) for load_id, route in loads.items()))
    final_state = tuple(((load_id, route[-1]) for load_id, route in loads.items()))
    movements = Graph()

    states = [(0, initial_state)]

    timer = Timer(timeout=timeout)
    solved = False

    while not solved:
        timer.timeout_check()

        score, state = states.pop(0)
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

                    new_score = sum(1 if a == b else 0 for a, b in zip(new_state, final_state))
                    if new_score < score:
                        continue
                    else:
                        states.append((new_score, new_state))
                        states = [(score, s) for score, s in states if score >= new_score]

                if final_state == new_state:
                    solved = True
                    break
        if not states:
            return None  # hill climbing doesn't lead to a solution.

    steps, best_path = movements.shortest_path(initial_state, final_state)
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



