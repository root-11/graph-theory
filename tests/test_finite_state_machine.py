from graph.finite_state_machine import FiniteStateMachine
from itertools import cycle


def test_traffic_light():
    green, yellow, red = 'Green', 'Yellow', 'Red'
    seq = cycle([green, yellow, red])
    _ = next(seq)
    fsm = FiniteStateMachine()
    fsm.add_transition(green, 'switch', yellow)
    fsm.add_transition(yellow, 'switch', red)
    fsm.add_transition(red, 'switch', green)
    fsm.set_initial_state(green)
    for _ in range(20):
        current_state = fsm.current_state
        new_state = next(seq)
        fsm.next('switch')
        assert fsm.current_state == new_state, (fsm.current_state, new_state)
        assert new_state != current_state, (new_state, current_state)


def test_turnstile():
    locked, unlocked = 'locked', 'unlocked'  # states
    push, coin = 'push', 'coin'  # actions
    fsm = FiniteStateMachine()
    fsm.add_transition(locked, coin, unlocked)
    fsm.add_transition(unlocked, push, locked)
    fsm.add_transition(locked, push, locked)
    fsm.add_transition(unlocked, coin, unlocked)
    try:
        assert fsm._initial_state_was_set is False
        fsm.next(coin)
        raise AssertionError
    except ValueError:
        pass

    try:
        assert fsm._initial_state_was_set is False
        fsm.set_initial_state('fish')
        raise AssertionError
    except ValueError:
        pass

    fsm.set_initial_state(locked)

    try:
        assert fsm._initial_state_was_set is True
        fsm.set_initial_state(locked)
        raise AssertionError
    except ValueError:
        assert fsm._initial_state_was_set is True
        pass

    # pay and go:
    display_state = set(fsm.options())
    assert display_state == {coin, push}, display_state
    assert fsm.current_state == locked
    fsm.next(action=coin)
    display_state = set(fsm.options())
    assert display_state == {coin, push}, display_state
    assert fsm.current_state == unlocked
    fsm.next(action=push)
    assert fsm.current_state == locked

    # try to cheat
    fsm.next(action=push)
    assert fsm.current_state == locked
    fsm.next(action=push)
    assert fsm.current_state == locked

    # pay and go:
    fsm.next(action=coin)
    assert fsm.current_state == unlocked
    fsm.next(action=push)
    assert fsm.current_state == locked

    try:
        assert fsm._initial_state_was_set is True
        fsm.next(action='fish')
        raise AssertionError
    except ValueError:
        pass

    fsm.add_transition(locked, 'fire', 'fire escape mode')
    fsm.next('fire')
    try:
        fsm.next(push)
        raise AssertionError
    except StopIteration:
        pass

