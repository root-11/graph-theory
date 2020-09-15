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


def dfs_resolve(graph, loads, timeout=None):  # <-- this is useless.
    """
    calculates the solution to the transshipment problem by
    search along a line of movements and backtracking when it
    no longer leads anywhere (DFS).

    :param graph network available for routing.
    :param loads: dictionary with load id and preferred route. Example:

        loads = {1: [1, 2, 3], 2: [3, 2, 1]}

    :param timeout: None, float or int timeout in milliseconds.

    """
    initial_state = tuple(((load_id, route[0]) for load_id, route in loads.items()))
    final_state = tuple(((load_id, route[-1]) for load_id, route in loads.items()))

    state = initial_state
    states = [initial_state]  # q
    path = []
    movements = Graph()
    visited = set()

    timer = Timer(timeout=timeout)

    while states:
        timer.timeout_check()

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
    """ generator for new states for DFS. """
    occupied = {i[1] for i in state}
    for load_id, location in state:
        options = (e for s, e, d in graph.edges(from_node=location) if e not in occupied)
        for option in options:
            new_state = tuple((lid, loc) if lid != load_id else (load_id, option) for lid, loc in state)
            if new_state in movements:
                continue
            movements.add_edge(state, new_state, 1)
            yield new_state


# collection of solution methods for the routing problem.
# insert, delete, append or substitute with your own methods as required.
methods = [
    bi_directional_progressive_bfs,  # <-- the fastest, but not always the best method.
    bi_directional_bfs,  # <-- best method so far.
    bfs_resolve,  # very slow, but will eventually find the best solution.
    # dfs_resolve,  <--- this simply doesn't work.
]

