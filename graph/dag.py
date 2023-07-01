from collections import deque
from .base import BasicGraph


def phase_lines(graph):
    """Determines the phase lines of a directed graph.
    This is useful for determining which tasks can be performed in 
    parallel. Each phase in the phaselines must be completed to assure
    that the tasks in the next phase can be performed with complete input.
    This is in contrast to Topological sort that only generates
    a queue of tasks, may be fine for a single processor, but has no 
    mechanism for coordination that all inputs for a task have been completed
    so that multiple processors can work on them.
    Example: DAG with tasks:
        u1      u4      u2      u3
        \       \       \_______\
        csg     cs3       append
        \       \           \
        op1     \           op3
        \       \           \
        op2     \           cs2
        \       \___________\
        cs1         join
        \           \
        map1        map2
        \___________\
            save
    phaselines = {
        "u1": 0, "u4": 0, "u2": 0, "u3": 0, 
        "csg": 1, "cs3": 1, "append": 1,
        "op1": 2, "op3": 2, "op2": 3, "cs2": 3,
        "cs1": 4, "join": 4,
        "map1": 5, "map2": 5,
        "save": 6,
    }  
    From this example it is visible that processing the 4 'uN' (uploads) is
    the highest degree of concurrency. This can be determined as follows:
        d = defaultdict(int)
        for _, pl in graph.phaselines():
            d[pl] += 1
        max_processors = max(d, key=d.get)
    :param graph: Graph
    :return: dictionary with node id : phase in cut.
    Note: To transform the phaselines into a task sequence use
    the following recipe:
    tasks = defaultdict(set)
    for node, phase in phaselines(graph):
        tasks[phase].add(node)
    To obtain a sort stable tasks sequence use:
    for phase in sorted(tasks):
        print(phase, list(sorted(tasks[phase]))
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")

    nmax = len(graph.nodes())
    phases = {n: nmax + 1 for n in graph.nodes()}
    phase_counter = 0

    g2 = graph.copy()
    q = list(g2.nodes(in_degree=0))  # Current iterations work queue
    if not q:
        raise AttributeError("The graph does not have any sources.")

    q2 = set()  # Next iterations work queue
    while q:
        for n1 in q:
            if g2.in_degree(n1)!=0:
                q2.add(n1)  # n1 has an in coming edge, so it has to wait.
                continue
            phases[n1] = phase_counter  # update the phaseline number
            for n2 in g2.nodes(from_node=n1):
                q2.add(n2)  # add node for next iteration

        # at this point the nodes with no incoming edges have been accounted
        # for, so now they may be removed for the working graph.
        for n1 in q:
            if n1 not in q2:
                g2.del_node(n1)  # remove nodes that have no incoming edges

        if set(q) == q2:
            raise AttributeError(f"Loop found: The graph is not acyclic!")

        # Finally turn the next iterations workqueue into current.
        # and increment the phaseline counter.
        q = [n for n in q2]
        q2.clear()
        phase_counter += 1
    return phases


def sources(graph, n):
    """Determines the set of all upstream sources of node 'n' in a DAG.
    :param graph: Graph
    :param n: node for which the sources are sought.
    :return: set of nodes
    """
    if not isinstance(graph, BasicGraph):
        raise TypeError(f"Expected subclass of BasicGraph, not {type(graph)}")
    if n not in graph:
        raise ValueError(f"{n} not in graph")

    nodes = {n}
    q = deque([n])
    while q:
        new = q.popleft()
        for src in graph.nodes(to_node=new):
            if src not in nodes:
                nodes.add(src)
            if src not in q:
                q.append(src)
    nodes.remove(n)
    return nodes