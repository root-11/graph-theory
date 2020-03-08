from graph import Graph
from itertools import count


class FiniteStateMachine(object):
    def __init__(self):
        self.states = Graph()
        self.current_state = None
        self._action_id = count()
        self.actions = {}
        self._initial_state_was_set = False

    def set_initial_state(self, state):
        if self._initial_state_was_set:
            raise ValueError(f"initial state has already been set.")
        if state not in self.states.nodes():
            raise ValueError(f"{state} is not a state.")
        self.current_state = state
        self._initial_state_was_set = True

    def add_transition(self, state_1, action, state_2):
        action_id = next(self._action_id)
        self.states.add_edge(node1=state_1, node2=action_id)
        self.states.add_edge(node1=action_id, node2=state_2)
        self.actions[action_id] = action

    def options(self):
        return [self.actions[i] for i in self.states.nodes(from_node=self.current_state)]

    def next(self, action):
        if self._initial_state_was_set is False:
            raise ValueError("initial state has not been set.")
        actions = self.states.nodes(from_node=self.current_state)
        if not actions:
            raise StopIteration("terminal node reached.")

        for action_id in actions:
            if self.actions[action_id] == action:
                self.current_state = self.states.nodes(from_node=action_id)[0]
                return
        raise ValueError(f"{self.current_state} does not permit {action}.")

