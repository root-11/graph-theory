from fractions import Fraction as F
from itertools import permutations, combinations_with_replacement

from graph import Graph
from tutorials.wtap import wtap


def test01_wtap():
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

    value, assignments = wtap(probabilities=g, weapons=weapons, target_values=target_values)
    assert isinstance(assignments, Graph)
    assert set(assignments.edges()) == {(2, 7, 0.1), (3, 6, 0.1), (1, 7, 0.1)}
    assert value == 16.07


def test02_wtap_with_fractional_probabilities():
    weapons = [1, 2, 3]
    probabilities = [
        (1, 5, F(1, 10)),
        (1, 6, F(1, 10)),
        (1, 7, F(1, 10)),
        (2, 5, F(1, 10)),
        (2, 6, F(1, 10)),
        (2, 7, F(1, 10)),
        (3, 5, F(1, 10)),
        (3, 6, F(1, 10)),
        (3, 7, F(1, 10))
    ]
    target_values = {5: 5, 6: 6, 7: 7}
    g = Graph(from_list=probabilities)

    value, assignments = wtap(probabilities=g, weapons=weapons, target_values=target_values)
    assert isinstance(assignments, Graph)
    assert set(assignments.edges()) == {(2, 7, F(1, 10)), (3, 6, F(1, 10)), (1, 7, F(1, 10))}
    assert float(value) == 16.07


def test03_wtap_from_wikipedia_all_permutations():
    g, weapons, target_values = wikipedia_wtap_setup()

    c = 0
    perfect_score = 4.95
    quality_score = 0
    quality_required = 0.97

    variations = {}
    damages = {}
    perms = set(permutations(weapons, len(weapons)))
    c = 0
    while perms:

        perm = perms.pop()
        perm2 = tuple(reversed(perm))
        perms.remove(perm2)
        damage1, ass1 = wtap(probabilities=g, weapons=list(perm), target_values=target_values)
        damage2, ass2 = wtap(probabilities=g, weapons=list(perm2), target_values=target_values)

        damageN = min(damage1, damage2)
        if damage1 == damageN:
            assignment = ass1
        else:
            assignment = ass2

        damage = wikipedia_wtap_damage_assessment(probabilities=g, assignment=assignment, target_values=target_values)
        assert round(damageN, 2) == round(damage, 2)
        damage = round(damage, 2)

        quality_score += damage
        c += 1

        if damage not in damages:
            s = "{:.3f} : {}".format(damage, wikipedia_wtap_pretty_printer(assignment))
            damages[damage] = s
        if damage not in variations:
            variations[damage] = 1
        else:
            variations[damage] += 1

    print("tested", c, "permutations. Found", len(damages), "variation(s)")
    if len(variations) > 1:
        for k, v in sorted(damages.items()):
            print(k, "frq: {}".format(variations[k]))

    solution_quality = perfect_score * c / quality_score
    if solution_quality >= quality_required:
        raise AssertionError("achieved {:.0f}%".format(solution_quality*100))
    print("achieved {:.0f}%".format(solution_quality * 100))


def test04_wtap_from_wikipedia_exhaustive():
    g, weapons, target_values = wikipedia_wtap_setup()

    best_result = sum(target_values.values())+1
    best_assignment = None
    c = 0
    for perm in permutations(weapons, len(weapons)):
        for combination in combinations_with_replacement([1, 2, 3], len(weapons)):
            L = [(w, t, g[w][t]) for w, t in zip(perm, combination)]
            a = Graph(from_list=L)
            r = wikipedia_wtap_damage_assessment(probabilities=g, assignment=a, target_values=target_values)
            if r < best_result:
                best_result = r
                best_assignment = L
            c += 1
    print(best_result, "is best result out of", c, "(exhaustive search)\n", best_assignment)
    assert best_result == 4.95, best_result


def wikipedia_wtap_setup():
    """
    A commander has 5 tanks, 2 aircraft and 1 sea vessel and is told to
    engage 3 targets with values 5,10,20 ...
    """
    tanks = ["tank-{}".format(i) for i in range(5)]
    aircrafts = ["aircraft-{}".format(i) for i in range(2)]
    ships = ["ship-{}".format(i) for i in range(1)]
    weapons = tanks + aircrafts + ships
    target_values = {1: 5, 2: 10, 3: 20}

    tank_probabilities = [
        (1, 0.3),
        (2, 0.2),
        (3, 0.5),
    ]

    aircraft_probabilities = [
        (1, 0.1),
        (2, 0.6),
        (3, 0.5),
    ]

    sea_vessel_probabilities = [
        (1, 0.4),
        (2, 0.5),
        (3, 0.4)
    ]

    category_and_probabilities = [
        (tanks, tank_probabilities),
        (aircrafts, aircraft_probabilities),
        (ships, sea_vessel_probabilities)
    ]

    probabilities = []
    for category, probs in category_and_probabilities:
        for vehicle in category:
            for prob in probs:
                probabilities.append((vehicle,) + prob)

    g = Graph(from_list=probabilities)
    return g, weapons, target_values


def wikipedia_wtap_damage_assessment(probabilities, assignment, target_values):
    assert isinstance(probabilities, Graph)

    assert isinstance(assignment, Graph)
    result = assignment.edges()
    assert isinstance(target_values, dict)

    survival_value = {}
    for item in result:
        weapon, target, damage = item
        if target not in survival_value:
            survival_value[target] = {}
        wtype = weapon.split("-")[0]
        if wtype not in survival_value[target]:
            survival_value[target][wtype] = 0
        survival_value[target][wtype] += 1

    total_survival_value = 0
    for target, assigned_weapons in survival_value.items():
        p = 1
        for wtype, quantity in assigned_weapons.items():
            weapon = wtype + "-0"
            p_base = (1 - probabilities[weapon][target])
            p *= p_base ** quantity

        total_survival_value += p * target_values[target]

    for target in target_values:
        if target not in survival_value:
            total_survival_value += target_values[target]

    return total_survival_value


def wikipedia_wtap_pretty_printer(assignment):
    assert isinstance(assignment, Graph)
    result = assignment.edges()
    survival_value = {}
    for item in result:
        weapon, target, damage = item
        if target not in survival_value:
            survival_value[target] = {}
        wtype = weapon.split("-")[0]
        if wtype not in survival_value[target]:
            survival_value[target][wtype] = 0
        survival_value[target][wtype] += 1

    L = []
    for target, wtypes in sorted(survival_value.items()):
        L.append("T-{}: ".format(target))
        _and = " + "
        for wtype, qty in sorted(wtypes.items()):
            if qty > 1:
                _wtype = wtype + "s"
            else:
                _wtype = wtype
            L.append("{} {}".format(qty, _wtype))
            L.append(_and)
        L.pop(-1)
        L.append(", ")
    s = "".join(L)
    return s


def test_to_verify_wikipedia_damage_assessment():
    g, weapons, target_values = wikipedia_wtap_setup()

    L = [
        ("tank-0", 1, 0.3),
        ("tank-1", 1, 0.3),
        ("tank-2", 1, 0.3),
        ("tank-3", 2, 0.2),
        ("tank-4", 2, 0.2),
        ("aircraft-0", 3, 0.5),
        ("aircraft-1", 3, 0.5),
        ("ship-0", 2, 0.5)

    ]
    assignment = Graph(from_list=L)
    assert 9.915 == wikipedia_wtap_damage_assessment(probabilities=g, assignment=assignment,
                                                     target_values=target_values)