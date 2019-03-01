


description = """ 
Example with a custom datastrcuture for nodes.
"""


class CustomNode(object):
    """ A sample 'Custom Node' for users to copy paste

    Example:

    applepie = set()
    apple_strudel = set()
    brownie = set()

    def distance(one, other):
        return 1 / len(one.intersect(other))

    G =


    """
    ids = 0  # FIXME.

    def __init__(self, values):
        """
        :param values: any object that should contain the values of the node
        """
        self.values = values
        CustomNode.ids += 1
        self._id = CustomNode.ids

    def __hash__(self):
        """
        :return: int, the node id used in all calculations.
        """
        return self._id