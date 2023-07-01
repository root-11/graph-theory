# Graph functions
# -----------------------------
from bisect import insort
from .base import BasicGraph
from collections import defaultdict


from heapq import heappop, heappush


def shortest_path(graph, start, end, avoids=None):
    """single source shortest path algorithm.
    :param graph: class Graph
    :param start: start node
    :param end: end node
    :param avoids: optional set,frozenset or list of nodes that cannot be a part of the path.
    :return distance, path (as list),
            returns float('inf'), [] if no path exists.
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    if start not in graph:
        raise ValueError(f"{start} not in graph")
    if end not in graph:
        raise ValueError(f"{end} not in graph")
    if avoids is None:
        visited = set()
    elif not isinstance(avoids, (frozenset, set, list)):
        raise TypeError(f"Expect obstacles as set or frozenset, not {type(avoids)}")
    else:
        visited = set(avoids)

    q, minimums = [(0, 0, start, ())], {start: 0}
    i = 1
    while q:
        (cost, _, v1, path) = heappop(q)
        if v1 not in visited:
            visited.add(v1)
            path = (v1, path)

            if v1 == end:  # exit criteria.
                L = []
                while path:
                    v, path = path[0], path[1]
                    L.append(v)
                L.reverse()
                return cost, L

            for _, v2, dist in graph.edges(from_node=v1):
                if v2 in visited:
                    continue
                prev = minimums.get(v2, None)
                next_node = cost + dist
                if prev is None or next_node < prev:
                    minimums[v2] = next_node
                    heappush(q, (next_node, i, v2, path))
                    i += 1
    return float("inf"), []


class ScanThread(object):
    __slots__ = ["cost", "n1", "path"]

    """ search thread for bidirectional search """

    def __init__(self, cost, n1, path=()):
        if not isinstance(path, tuple):
            raise TypeError(f"Expected a tuple, not {type(path)}")
        self.cost = cost
        self.n1 = n1
        self.path = path

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return f"{self.cost}:{self.path}"


class BiDirectionalSearch(object):
    """data structure for organizing bidirectional search"""

    forward = True
    backward = False

    def __str__(self):
        if self.forward == self.direction:
            return "forward scan"
        return "backward scan"

    def __init__(self, graph, start, direction=True, avoids=None):
        """
        :param graph: class Graph.
        :param start: first node in the search.
        :param direction: bool
        :param avoids: nodes that cannot be a part of the solution.
        """
        if not isinstance(graph, BasicGraph):
            raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
        if start not in graph:
            raise ValueError("start not in graph.")
        if not isinstance(direction, bool):
            raise TypeError(f"Expected boolean, not {type(direction)}")
        if avoids is None:
            self.avoids = frozenset()
        elif not isinstance(avoids, (frozenset, set, list)):
            raise TypeError(f"Expect obstacles as set or frozenset, not {type(avoids)}")
        else:
            self.avoids = frozenset(avoids)

        self.q = []
        self.q.append(ScanThread(cost=0, n1=start))
        self.graph = graph
        self.boundary = set()  # visited.
        self.mins = {start: 0}
        self.paths = {start: ()}
        self.direction = direction
        self.sp = ()
        self.sp_length = float("inf")

    def update(self, sp, sp_length):
        if sp_length > self.sp_length:
            raise ValueError("Bad logic!")
        self.sp = sp
        self.sp_length = sp_length

    def search(self, other):
        assert isinstance(other, BiDirectionalSearch)
        if not self.q:
            return

        sp, sp_length = self.sp, self.sp_length

        st = self.q.pop(0)
        assert isinstance(st, ScanThread)
        if st.cost > self.sp_length:
            return

        self.boundary.add(st.n1)

        if st.n1 in other.boundary:  # if there's an intercept between the two searches ...
            if st.cost + other.mins[st.n1] < self.sp_length:
                sp_length = st.cost + other.mins[st.n1]
                if self.direction == self.forward:
                    sp = tuple(reversed(st.path)) + (st.n1,) + other.paths[st.n1]
                else:  # direction == backward:
                    sp = tuple(reversed(other.paths[st.n1])) + (st.n1,) + st.path

                self.q = [a for a in self.q if a.cost < sp_length]

        if self.direction == self.forward:
            edges = sorted((d, e) for s, e, d in self.graph.edges(from_node=st.n1) if e not in self.avoids)
        else:
            edges = sorted((d, s) for s, e, d in self.graph.edges(to_node=st.n1) if s not in self.avoids)

        for dist, n2 in edges:
            n2_dist = st.cost + dist
            if n2_dist > self.sp_length:  # no point pursuing as the solution is worse.
                continue
            if n2 in other.mins and n2_dist + other.mins[n2] > self.sp_length:  # already longer than lower bound.
                continue

            # at this point we can't dismiss that n2 will lead to a better solution, so we retain it.
            prev = self.mins.get(n2, None)
            if prev is None or n2_dist < prev:
                self.mins[n2] = n2_dist
                path = (st.n1,) + st.path
                self.paths[n2] = path
                insort(self.q, ScanThread(n2_dist, n2, path))

        self.update(sp, sp_length)
        other.update(sp, sp_length)


def shortest_path_bidirectional(graph, start, end, avoids=None):
    """Bidirectional search using lower bound.
    :param graph: Graph
    :param start: start node
    :param end: end node
    :param avoids: nodes that cannot be a part of the shortest path.
    :return: shortest path
    In Section 3.4.6 of Artificial Intelligence: A Modern Approach, Russel and
    Norvig write:
    Bidirectional search is implemented by replacing the goal test with a check
    to see whether the frontiers of the two searches intersect; if they do,
    a solution has been found. It is important to realize that the first solution
    found may not be optimal, even if the two searches are both breadth-first;
    some additional search is required to make sure there isn't a shortcut
    across the gap.
    To overcome this limit for weighted graphs, I've added a lower bound, so
    that when the two searches intersect, the lower bound is updated and the
    lower bound path is stored. In subsequent searches any path shorter than
    the lower bound, leads to an update of the lower bound and shortest path.
    The algorithm stops when all nodes on the frontier exceed the lower bound.
    ----------------
    The algorithms works as follows:
    Lower bound = float('infinite')
    shortest path = None
    Two queues (forward scan and backward scan) are initiated with respectively
    the start and end node as starting point for each scan.
    while there are nodes in the forward- and backward-scan queues:
        1. select direction from (forward, backward) in alternations.
        2. pop the top item from the queue of the direction.
           (The top item contains the node N that is nearest the starting point for the scan)
        3. Add the node N to the scan-directions frontier.
        4. If the node N is in the other directions frontier:
            the path distance from the directions _start_ to the _end_ via
            the point of intersection (N), is compared with the lower bound.
            If the path distance is less than the lower bound:
            *lower bound* is updated with path distance, and,
            the *shortest path* is recorded.
        5. for each node N2 (to N if backward, from N if forward):
            the distance D is accumulated
            if D > lower bound, the node N2 is ignored.
            if N2 is within the other directions frontier and D + D.other > lower bound, the node N2 is ignored.
            otherwise:
                the path P recorded
                and N2, D and P are added to the directions scan queue
    The algorithm terminates when the scan queues are exhausted.
    ----------------
    Given that:
    - `c` is the connectivity of the nodes in the graph,
    - `R` is the length of the path,
    the explored solution landscape can be estimated as:
    A = c * (R**2), for single source shortest path
    A = c * 2 * (1/2 * R) **2, for bidirectional shortest path
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    if start not in graph:
        raise ValueError("start not in graph.")
    if end not in graph:
        raise ValueError("end not in graph.")

    forward = BiDirectionalSearch(graph, start=start, direction=BiDirectionalSearch.forward, avoids=avoids)
    backward = BiDirectionalSearch(graph, start=end, direction=BiDirectionalSearch.backward, avoids=avoids)

    while any((forward.q, backward.q)):
        forward.search(other=backward)
        backward.search(other=forward)

    return forward.sp_length, list(forward.sp)


class ShortestPathCache(object):
    """
    Data structure optimised for repeated calls to shortest path.
    Used by shortest path when using keyword `memoize=True`
    """

    def __init__(self, graph):
        if not isinstance(graph, BasicGraph):
            raise TypeError(f"Expected type Graph, not {type(graph)}")
        self.graph = graph
        self.cache = {}
        self.repeated_cache = {}

    def _update_cache(self, path):
        """private method for updating the cache for future lookups.
        :param path: tuple of nodes
        Given a shortest path, all steps along the shortest path,
        also constitute the shortest path between each pair of steps.
        """
        if not isinstance(path, (list, tuple)):
            raise TypeError
        b = len(path)
        if b < 2:
            return
        if b == 2:
            dist = self.graph.distance_from_path(path)
            self.cache[(path[0], path[-1])] = (dist, tuple(path))

        for a, _ in enumerate(path):
            section = tuple(path[a : b - a])
            if len(section) < 3:
                break
            dist = self.graph.distance_from_path(section)
            self.cache[(section[0], section[-1])] = (dist, section)

        for ix, start in enumerate(path[1:-1]):
            section = tuple(path[ix:])
            dist = self.graph.distance_from_path(section)
            self.cache[(section[0], section[-1])] = (dist, section)

    def shortest_path(self, start, end, avoids=None):
        """Shortest path method that utilizes caching and bidirectional search"""
        if start not in self.graph:
            raise ValueError("start not in graph.")
        if end not in self.graph:
            raise ValueError("end not in graph.")
        if avoids is None:
            pass
        elif not isinstance(avoids, (frozenset, set, list)):
            raise TypeError(f"Expect obstacles as None, set or frozenset, not {type(avoids)}")
        else:
            avoids = frozenset(avoids)

        d = 0 if start == end else None
        p = ()

        if isinstance(avoids, frozenset):
            # as avoids can be volatile, it is not possible to benefit from the caching
            # methodology. This does however not mean that we need to forfeit the benefit
            # of bidirectional search.
            hash_key = hash(avoids)
            d, p = self.repeated_cache.get((start, end, hash_key), (None, None))
            if d is None:
                d, p = shortest_path_bidirectional(self.graph, start, end, avoids=avoids)
                self.repeated_cache[(start, end, hash_key)] = (d, p)
        else:
            if d is None:  # is it cached?
                d, p = self.cache.get((start, end), (None, None))

            if d is None:  # search for it.
                _, p = shortest_path_bidirectional(self.graph, start, end)
                self._update_cache(p)
                d, p = self.cache[(start, end)]

        return d, list(p)


class SPLength(object):
    def __init__(self):
        self.value = float("inf")


def distance_from_path(graph, path):
    """Calculates the distance for the path in graph
    :param graph: class Graph
    :param path: list of nodes
    :return: distance
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    if not isinstance(path, (tuple, list)):
        raise TypeError(f"expected tuple or list, not {type(path)}")

    cache = defaultdict(dict)
    path_length = 0
    for idx in range(len(path) - 1):
        n1, n2 = path[idx], path[idx + 1]

        # if the edge exists...
        d = graph.edge(n1, n2, default=None)
        if d:
            path_length += d
            continue

        # if we've seen the edge before...
        d = cache.get((n1, n2), None)
        if d:
            path_length += d
            continue

        # if there no alternative ... (search)
        d, _ = shortest_path(graph, n1, n2)
        if d == float("inf"):
            return float("inf")  # <-- Exit if there's no path.
        else:
            cache[(n1, n2)] = d
        path_length += d
    return path_length
