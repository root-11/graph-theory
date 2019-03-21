from graph import Graph
from uuid import uuid4
from itertools import permutations

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


def assignment_problem(agents_and_tasks, agents, tasks):
    """ The assignment problem solver expects a bi-partite graph
    with agents, tasks and the value/cost of each task, as links,
    so that the relationship is explicit as:

        G[agent 1][task 1] = value.

    The optimal assignment is determined as an alternating auction
    (see Dmitri Bertsekas, MIT) which maximises the value.
    Once all agents are assigned the alternating auction halts.

    :param agents_and_tasks: Graph
    :param agents: List of nodes that are agents.
    :param tasks: List of nodes that are tasks.
    :return: optimal assignment as list of edges (agent, task, value)
    """
    assert isinstance(agents_and_tasks, Graph)
    assert isinstance(agents, list)
    assert isinstance(tasks, list)

    unassigned_agents = agents
    v_null = min(v for a, t, v in agents_and_tasks.edges()) - 1

    dummy_tasks = set()
    if len(agents) > len(tasks):  # make dummy tasks.
        dummy_tasks_needed = len(agents) - len(tasks)

        for i in range(dummy_tasks_needed):
            task = uuid4().hex
            dummy_tasks.add(task)
            tasks.append(task)
            for agent in agents:
                agents_and_tasks.add_edge(agent, task, v_null)
        v_null -= 1

    unassigned_tasks = set(tasks)
    assignments = Graph()

    while unassigned_agents:
        n = unassigned_agents.pop(0)
        # select phase:
        value_and_task_for_n = [(v, t) for a, t, v in agents_and_tasks.edges() if a == n]
        value_and_task_for_n.sort(reverse=True)

        for v, t in value_and_task_for_n:  # for each opportunity (in ranked order)
            v_n2_t = v_null
            if t in assignments:  # if connected get whoever it is connected to.
                for n2 in assignments[t]:  # and lookup the value.
                    v_n2_t = assignments[t][n2]
                    break
            if v > v_n2_t:  # if the opportunity is better.
                if t in assignments:  # and if there is a previous relationship.
                    unassigned_agents.append(n2)  # add the removed node to unassigned.
                    del assignments[t]  # erase any previous relationship.
                else:
                    unassigned_tasks.remove(t)
                assignments.add_edge(t, n, v)  # record the new relationship.
                break

    return [(a, t, v) for t, a, v in assignments.edges() if t not in dummy_tasks]


def test_01_taxis_and_customers():
    """ Test where taxis are assigned to customers, so that the
    time to travel is minimised.

    Note that the travel time is negative, whereby the optimal
    assignment will have the minimal reponse time for all taxis
    to collect all customers.

    """
    taxis = [1, 2, 3]
    customers = [4, 5, 6]
    # relationships with distance as minutes apart.
    L = [
        (1, 4, -11),  # taxi 1, customer 4, 11 minutes apart.
        (1, 5, -23),
        (1, 6, -33),
        (2, 4, -14),
        (2, 5, -17),
        (2, 6, -34),
        (3, 4, -22),
        (3, 5, -19),
        (3, 6, -13)
    ]
    relationships = Graph(from_list=L)

    permutations_checked = 0
    for taxis_ in permutations(taxis, len(taxis)):
        for customers_ in permutations(customers, len(customers)):
            permutations_checked += 1
            print(permutations_checked, taxis_, customers_)
            assignment = assignment_problem(agents_and_tasks=relationships,
                                            agents=list(taxis_),
                                            tasks=list(customers_))
            assert set(assignment) == {(1, 4, -11), (2, 5, -17), (3, 6, -13)}
            assert sum(v for a, t, v in assignment) == sum([-11, -17, -13])
    print("The assignment problem solver is insensitive to initial conditions.")


def test_02_taxis_and_more_customers():
    """
    Like test_01, but with an additional customer who is attractive
    (value = -10)

    The same conditions exist as in test_01, but the least attractice
    customer (5) with value -17 is expected to be dropped.
    """
    taxis = [1, 2, 3]
    customers = [4, 5, 6, 7]
    # relationships with distance as minutes apart.
    L = [
        (1, 4, -11),  # taxi 1, customer 4, 11 minutes apart.
        (1, 5, -23),
        (1, 6, -33),
        (1, 7, -10),
        (2, 4, -14),
        (2, 5, -17),
        (2, 6, -34),
        (2, 7, -10),
        (3, 4, -22),
        (3, 5, -19),
        (3, 6, -13),
        (3, 7, -10)
    ]
    relationships = Graph(from_list=L)
    assignment = assignment_problem(agents_and_tasks=relationships,
                                    agents=list(taxis),
                                    tasks=list(customers))
    assert set(assignment) == {(1, 7, -10), (2, 4, -14), (3, 6, -13)}
    assert sum(v for a, t, v in assignment) > sum([-11, -17, -13])


def test_03_taxis_but_fewer_customers():
    """
    Like test_01, but with an additional taxi who is more attractive.

    We hereby expect that the new taxi (7) will steal the customer
    from 2.

    """
    taxis = [1, 2, 3, 7]
    customers = [4, 5, 6]
    # relationships with distance as minutes apart.
    L = [
        (1, 4, -11),  # taxi 1, customer 4, 11 minutes apart.
        (1, 5, -23),
        (1, 6, -33),
        (2, 4, -14),
        (2, 5, -17),
        (2, 6, -34),
        (3, 4, -22),
        (3, 5, -19),
        (3, 6, -13),
        (7, 4, -11),
        (7, 5, -11),
        (7, 6, -11),
    ]
    relationships = Graph(from_list=L)
    assignment = assignment_problem(agents_and_tasks=relationships,
                                    agents=list(taxis),
                                    tasks=list(customers))
    assert set(assignment) == {(1, 4, -11), (2, 5, -17), (7, 6, -11)}
    assert sum(v for a, t, v in assignment) > sum([-11, -17, -13])