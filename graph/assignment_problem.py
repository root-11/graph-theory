from uuid import uuid4

from graph import Graph

__all__ = ['ap_solver', 'wtap_solver']


def ap_solver(graph):
    """

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

    ----------------------------------------------------

    The assignment problem solver expects a bi-partite graph
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


def wtap_solver(probabilities, weapons, target_values):
    """
    Definition:

    Weapons target assignment problem (WTAP)

    The problem instance has a number weapons which can be assigned
    to engage targets, with a success rate of P(x). Targets have values V.
    If a weapon is engaged against a target, and is successful, the value is
    reduced to zero. Expected outcome of an engagement (D,T) is thereby

        O = V * (1-P(x))

    The optimal assignment is minimises the value of the targets.

        min E( O )

    for more see: https://en.wikipedia.org/wiki/Weapon_target_assignment_problem

    Variations:

    - if V is unknown, use v = 1. This maximises the exploitation of the available
    probabilities.

    Method used:

    1. initial assignment using greedy algorithm;
    2. followed by search for improvements.

    ----------------

    :param probabilities: instance of Graph, where the relationship
    between weapons and targets is given as the probability to a
    successful engagement of the device.
    :param weapons: list of devices.
    :param target_values: dict , where the d[target] = value of target.
    :return: tuple: value of target after attack, optimal assignment

    """
    assert isinstance(probabilities, Graph)
    assert isinstance(weapons, list)
    assert isinstance(target_values, dict)
    # first: verify internal integrity of inputs.
    weaponset = set(weapons)
    targetset = set(target_values)
    id_overlap = weaponset.intersection(targetset)
    if id_overlap:
        raise ValueError(f"weapon ids in target_values for {id_overlap}")
    target_prob_set = {e for s, e, d in probabilities.edges()}
    if targetset > target_prob_set:
        raise ValueError(f"targets have no probabilities: {targetset-target_prob_set}")

    # second: clear the memory from the validations.
    weaponset.clear()
    targetset.clear()
    target_prob_set.clear()

    # then: Calculate the solution.
    assignments = Graph()
    current_target_values = sum(target_values.values()) + 1

    improvements = {}
    while True:
        for w in weapons:
            # calculate the effect of engaging in all targets.
            effect_of_assignment = {}
            for _, t, p in probabilities.edges(from_node=w):
                current_engagement = _get_current_engagement(w, assignments)
                if current_engagement != t:
                    if w in assignments and current_engagement is not None:
                        assignments.del_edge(w, current_engagement)
                    assignments.add_edge(w, t, value=probabilities.edge(w, t))
                effect_of_assignment[t] = _damages(probabilities=probabilities,
                                                   assignment=assignments,
                                                   target_values=target_values)

            damage_and_targets = [(v, t) for t, v in effect_of_assignment.items()]
            damage_and_targets.sort()
            best_alt_damage, best_alt_target = damage_and_targets[0]
            nett_effect = current_target_values - best_alt_damage
            improvements[w] = max(0, nett_effect)

            current_engagement = _get_current_engagement(w, assignments)
            if current_engagement != best_alt_target:
                if w in assignments and current_engagement is not None:
                    assignments.del_edge(w, current_engagement)
                assignments.add_edge(w, best_alt_target, probabilities.edge(w, best_alt_target))
            current_target_values = effect_of_assignment[best_alt_target]
        if sum(improvements.values()) == 0:
            break
    return current_target_values, assignments


def _get_current_engagement(d, assignment):
    """ helper for WTAP solver
    Calculates the current engagement
    :param d: device
    :param assignment: class Graph.
    :return:
    """
    if d in assignment:
        for d, t, v in assignment.edges(from_node=d):
            return t
    return None


def _damages(probabilities, assignment, target_values):
    """ helper for WTAP solver
    :param probabilities: graph with probability of device effect on target
    :param assignment: graph with links between device and targets.
    :param target_values: dict with [target]=value.
    :return: total survival value.
    """
    assert isinstance(probabilities, Graph)
    assert isinstance(assignment, Graph)
    assert isinstance(target_values, dict)

    survival_value = {target: [] for target in target_values}
    for edge in assignment.edges():
        weapon, target, damage = edge

        p = probabilities.edge(weapon, target)
        survival_value[target].append(p)

    total_survival_value = 0
    for target, assigned_probabilities in survival_value.items():
        p = 1
        for p_ in assigned_probabilities:
            p *= (1 - p_)
        total_survival_value += p * target_values[target]

    return total_survival_value