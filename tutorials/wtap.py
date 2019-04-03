from graph import Graph

__description__ = """

Definition:

Weapons target assignment problem (WTAP)

The problem instance has a number weapons which can be assigned
to engage targets, with a success rate of P(x). Targets have values V.
If a weapon is engaged against a target, and is successful, the value is 
reduced to zero. Expected outcome of an engagement (D,T) is thereby

    O = V * (1-P(x))

The optimal assignment is minimises the value of the targets.

    min E( O )

[1].

Variations:

- if V is unknown, use v = 1. This maximises the exploitation of the available
probabilities.

Solution methods:

1. Dynamic programming problem.
2. Alternating iterative auction.


[1] https://en.wikipedia.org/wiki/Weapon_target_assignment_problem
"""


__all__ = ["wtap"]


def wtap(probabilities, weapons, target_values):
    """
    :param probabilities: instance of Graph, where the relationship
    between weapons and targets is given as the probability to a
    successful engagement of the device.
    :param weapons: list of devices.
    :param target_values: dict , where the d[target] = value of target.
    :return: tuple: value of target after attack, optimal assignment

    Method:

    1. initial assignment using greedy algorithm;
    2. followed by search for improvements.

    """
    assert isinstance(probabilities, Graph)
    assert isinstance(weapons, list)
    assert isinstance(target_values, dict)

    assignments = Graph()
    current_target_values = sum(target_values.values())+1

    improvements = {}
    while True:
        for w in weapons:
            # calculate the effect of engaging in all targets.
            effect_of_assignment = {}
            for t, p in probabilities[w].items():
                current_engagement = get_current_engagement(w, assignments)
                if current_engagement != t:
                    if w in assignments and current_engagement is not None:
                        del assignments[w][current_engagement]
                    assignments.add_edge(w, t, probabilities[w][t])
                effect_of_assignment[t] = damages(probabilities=probabilities, assignment=assignments, target_values=target_values)

            damage_and_targets = [(v, t) for t, v in effect_of_assignment.items()]
            damage_and_targets.sort()
            best_alt_damage, best_alt_target = damage_and_targets[0]
            nett_effect = current_target_values - best_alt_damage
            improvements[w] = max(0, nett_effect)

            current_engagement = get_current_engagement(w, assignments)
            if current_engagement != best_alt_target:
                if w in assignments and current_engagement is not None:
                    del assignments[w][current_engagement]
                assignments.add_edge(w, best_alt_target, probabilities[w][t])
            current_target_values = effect_of_assignment[best_alt_target]
        if sum(improvements.values()) == 0:
            break
    return current_target_values, assignments


def get_current_engagement(d, assignment):
    """
    Calculates the current engagement
    :param d: device
    :param assignment: class Graph.
    :return:
    """
    if d in assignment:
        for t in assignment[d]:
            return t
    return None


def damages(probabilities, assignment, target_values):
    """
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

        p = probabilities[weapon][target]
        survival_value[target].append(p)

    total_survival_value = 0
    for target, assigned_probabilities in survival_value.items():
        p = 1
        for p_ in assigned_probabilities:
            p *= (1-p_)
        total_survival_value += p * target_values[target]

    return total_survival_value