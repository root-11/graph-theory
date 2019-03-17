from graph import Graph
from fractions import Fraction

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


def wtap(probabilities, weapons, target_values):
    """
    :param probabilities: instance of Graph, where the relationship
    between weapons and targets is given as the probability to a
    successful engagement of the device.
    :param weapons: list of devices.
    :param target_values: dict , where the d[target] = value of target.
    :return: optimal assignment

    Method:

    1. initial assignment using greedy algorithm;
    2. followed by search for improvements.

    """

    def get_current_engagement(d, assignment):
        if d in assignment:
            for t in assignment[d]:
                return t
        return None

    assert isinstance(probabilities, Graph)
    assert isinstance(weapons, list)
    assert isinstance(target_values, dict)

    unassigned_devices = weapons[:]
    assignments = Graph()
    cumulative_survival_prob = {t: 1 for t in target_values}  # d[target] = probability of survival.

    improvements = {}
    # initial assignment using greedy algorithm
    while unassigned_devices or sum(improvements.values()) > 0:
        if not unassigned_devices:
            unassigned_devices = weapons[:]
        d = unassigned_devices.pop(0)
        damage = [(p * target_values[t], t) for t, p in probabilities[d].items()]
        damage.sort(reverse=True)  # most damage at the top of the list.

        # calculate the effect of engaging in all targets.
        current_engagement = get_current_engagement(d, assignments)

        effect_of_assignment = {}

        for v, t in damage:
            if t is current_engagement:
                p = cumulative_survival_prob[t]
            else:
                p = cumulative_survival_prob[t] * (1 - probabilities[d][t])  # Bayes!
            effect_of_assignment[t] = (p - cumulative_survival_prob[t]) * target_values[t]

        min_values = [(v, t) for t, v in effect_of_assignment.items()]
        min_values.sort()
        damage_effect, best_target = min_values[0]

        if current_engagement is None:
            assignments.add_edge(d, best_target, damage_effect)
            cumulative_survival_prob[best_target] *= (1 - probabilities[d][best_target])
            improvements[d] = abs(damage_effect)

        elif current_engagement != best_target:
            current_damage_effect = assignments[d][current_engagement]
            if abs(damage_effect) > abs(current_damage_effect):
                del assignments[d][current_engagement]
                cumulative_survival_prob[best_target] /= (1 - probabilities[d][best_target])

                assignments.add_edge(d, best_target, damage_effect)
                cumulative_survival_prob[best_target] *= (1 - probabilities[d][best_target])
                improvements[d] = abs(damage_effect)
            else:
                improvements[d] = 0
        elif current_engagement == best_target:
            improvements[d] = 0
        else:
            raise Exception("!")

    return assignments


def test01_weapons_target_assignment_problem():
    weapons = [1, 2, 3]
    probabilities = [
        (1, 5, 0.1),
        (1, 6, 0.1),
        (1, 7, 0.1),
        (2, 5, 0.1),
        (2, 6, 0.1),
        (2, 7, 0.1),
        (3, 5, 0.1),
        (3, 6, 0.1),
        (3, 7, 0.1)
    ]
    target_values = {5: 5, 6: 6, 7: 7}
    g = Graph(from_list=probabilities)

    assignments = wtap(probabilities=g, weapons=weapons, target_values=target_values)
    assert isinstance(assignments, Graph)
    assert set(assignments.edges()) == {(1, 7, -0.6999999999999998), (2, 7, -0.6299999999999998), (3, 6, -0.5999999999999999)}

