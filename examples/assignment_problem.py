from graph import Graph
from uuid import uuid4

__description__ = """

ASSIGNMENT PROBLEM

Definition:

The problem instance has a number of agents and a number of tasks. 
Any agent can be assigned to perform any task, incurring some cost that may 
vary depending on the agent-task assignment. It is required to perform all 
tasks by assigning exactly one agent to each task and exactly one task to each 
agent in such a way that the total cost of the assignment is minimized.[1]

Variations:

- If there are more agents than tasks the problem can be solved by creating a
"do nothing tasks" with a cost of zero. The assignment problem solver does this
automatically.

- If there are more tasks than agents then the problem is a knapsack problem. 
The assignment problem solver handles this case gracefully too.


Solution methods:

1. Using maximum flow method.
2. Using alternating iterative auction.

[1] https://en.wikipedia.org/wiki/Assignment_problem

"""


def assignment_problem(graph):
    """ The assignment problem solver expects a bi-partite graph
    with agents, tasks and the value/cost of each task, as links,
    so that the relationship is explicit as:

        value = g.edge(agent 1, task 1)

    The optimal assignment is determined as an alternating auction
    (see Dmitri Bertsekas, MIT) which maximises the value.
    Once all agents are assigned the alternating auction halts.

    :param graph: Graph
    :return: optimal assignment as list of edges (agent, task, value)
    """
    assert isinstance(graph, Graph)
    agents = [n for n in graph.nodes(in_degree=0)]
    tasks = [n for n in graph.nodes(out_degree=0)]

    unassigned_agents = agents
    v_null = min(v for a, t, v in graph.edges()) - 1

    dummy_tasks = set()
    if len(agents) > len(tasks):  # make dummy tasks.
        dummy_tasks_needed = len(agents) - len(tasks)

        for i in range(dummy_tasks_needed):
            task = uuid4().hex
            dummy_tasks.add(task)
            tasks.append(task)
            for agent in agents:
                graph.add_edge(agent, task, v_null)
        v_null -= 1

    unassigned_tasks = set(tasks)
    assignments = Graph()

    while unassigned_agents:
        n = unassigned_agents.pop(0)  # select phase:
        value_and_task_for_n = [(v, t) for a, t, v in graph.edges(from_node=n)]
        value_and_task_for_n.sort(reverse=True)
        for v, t in value_and_task_for_n:  # for each opportunity (in ranked order)
            d = v_null
            for s, e, d in assignments.edges(from_node=t):  # if connected, get whoever it is connected to.
                break

            if v > d:  # if the opportunity is better.
                if t in assignments:  # and if there is a previous relationship.
                    unassigned_agents.append(e)  # add the removed node to unassigned.
                    assignments.del_edge(t, e)  # erase any previous relationship.
                else:
                    unassigned_tasks.remove(t)
                assignments.add_edge(t, n, v)  # record the new relationship.
                break

    return [(a, t, v) for t, a, v in assignments.edges() if t not in dummy_tasks]

