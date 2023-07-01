from .base import BasicGraph
from .dag import phase_lines
from .topological_sort import topological_sort


class Task(object):
    """Helper for critical path method"""

    __slots__ = ["task_id", "duration", "earliest_start", "earliest_finish", "latest_start", "latest_finish"]

    def __init__(
        self,
        task_id,
        duration,
        earliest_start=0,
        latest_start=0,
        earliest_finish=float("inf"),
        latest_finish=float("inf"),
    ):
        self.task_id = task_id
        self.duration = duration
        self.earliest_start = earliest_start
        self.latest_start = latest_start
        self.earliest_finish = earliest_finish
        self.latest_finish = latest_finish

    def __eq__(self, other):
        if not isinstance(other, Task):
            raise TypeError(f"can't compare {type(other)} with {type(self)}")
        if any(
            (
                self.task_id != other.task_id,
                self.duration != other.duration,
                self.earliest_start != other.earliest_start,
                self.latest_start != other.latest_start,
                self.earliest_finish != other.earliest_finish,
                self.latest_finish != other.latest_finish,
            )
        ):
            return False
        return True

    @property
    def slack(self):
        return self.latest_finish - self.earliest_finish

    @property
    def args(self):
        return (
            self.task_id,
            self.duration,
            self.earliest_start,
            self.latest_start,
            self.earliest_finish,
            self.latest_finish,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}{self.args}"

    def __str__(self):
        return self.__repr__()


def critical_path(graph):
    """
    The critical path method determines the
    schedule of a set of project activities that results in
    the shortest overall path.
    :param graph: acyclic graph where:
        nodes are task_id and node_obj is a value
        edges determines dependencies
        (see example below)
    :return: critical path length, schedule.
        schedule is list of Tasks
    Recipes:
    (1) Setting up the graph:
        tasks = {'A': 10, 'B': 20, 'C': 5, 'D': 10, 'E': 20, 'F': 15, 'G': 5, 'H': 15}
        dependencies = [
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'D'),
            ('D', 'E'),
            ('A', 'F'),
            ('F', 'G'),
            ('G', 'E'),
            ('A', 'H'),
            ('H', 'E'),
        ]
        g = Graph()
        for task, duration in tasks.items():
            g.add_node(task, obj=duration)
        for n1, n2 in dependencies:
            g.add_edge(n1, n2, 0)
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")

    # 1. A topologically sorted list of nodes is prepared (topo.order).
    order = list(topological_sort(graph))  # this will raise if there are loops.

    d = {}
    critical_path_length = 0

    for task_id in order:  # 2. forward pass:
        predecessors = [d[t] for t in graph.nodes(to_node=task_id)]
        duration = graph.node(task_id)
        if not isinstance(duration, (float, int)):
            raise ValueError(f"Expected task {task_id} to have a numeric duration, but got {type(duration)}")
        t = Task(task_id=task_id, duration=duration)
        # 1. the earliest start and earliest finish is determined in topological order.
        t.earliest_start = max(t.earliest_finish for t in predecessors) if predecessors else 0
        t.earliest_finish = t.earliest_start + t.duration
        d[task_id] = t
        # 2. the path length is recorded.
        critical_path_length = max(t.earliest_finish, critical_path_length)

    for task_id in reversed(order):  # 3. backward pass:
        successors = [d[t] for t in graph.nodes(from_node=task_id)]
        t = d[task_id]
        # 1. the latest start and finish is determined in reverse topological order
        t.latest_finish = min(t.latest_start for t in successors) if successors else critical_path_length
        t.latest_start = t.latest_finish - t.duration

    return critical_path_length, d


def critical_path_minimize_for_slack(graph):
    """
    Determines the critical path schedule and attempts to minimise
    the concurrent resource requirements by inserting the minimal
    number of artificial dependencies.
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")

    cpl, schedule = critical_path(graph)
    phases = phase_lines(graph)

    new_graph = graph.copy()
    slack_nodes = [t for k, t in schedule.items() if t.slack > 0]  # slack == 0 is on the critical path.
    slack_nodes.sort(reverse=True, key=lambda t: t.slack)  # most slack on top as it has more freedom to move.
    slack_node_ids = [t.task_id for t in slack_nodes]
    slack = sum(t.slack for t in schedule.values())

    # Part 1. Use a greedy algorithm to determine an initial solution.
    new_edges = []
    stop = False
    for node in sorted(slack_nodes, reverse=True, key=lambda t: t.duration):
        if stop:
            break
        # identify any option for insertion:
        n1 = node.task_id
        for n2 in (
            n2
            for n2 in slack_node_ids
            if n2 != n1  # ...not on critical path
            and phases[n2] >= phases[n1]  # ...not pointing to the source
            and new_graph.edge(n2, n1) is None  # ...downstream
            and new_graph.edge(n1, n2) is None  # ... not creating a cycle.
        ):  # ... not already a dependency.
            new_graph.add_edge(n1, n2)
            new_edges.append((n1, n2))

            cpl2, schedule2 = critical_path(new_graph)
            if cpl2 != cpl:  # the critical path is not allowed to be longer. Abort.
                new_graph.del_edge(n1, n2)
                new_edges.remove((n1, n2))
                continue

            slack2 = sum(t.slack for t in schedule2.values())

            if slack2 < slack and cpl == cpl2:  # retain solution!
                slack = slack2
                if slack == 0:  # an optimal solution has been found!
                    stop = True
                    break
            else:
                new_graph.del_edge(n1, n2)
                new_edges.remove((n1, n2))

    # Part 2. Purge non-effective edges.
    for edge in new_graph.edges():
        n1, n2, _ = edge
        if graph.edge(n1, n2) is not None:
            continue  # it's an original edge.
        # else: it's an artificial edge:
        new_graph.del_edge(n1, n2)
        cpl2, schedule2 = critical_path(new_graph)
        slack2 = sum(t.slack for t in schedule2.values())

        if cpl2 == cpl and slack2 == slack:
            continue  # the removed edge had no effect and just made the graph more complicated.
        else:
            new_graph.add_edge(n1, n2)

    return new_graph