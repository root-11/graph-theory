

description = """ 
Example with a custom datastrcuture for nodes.
"""


class CustomNode(object):
    """ A sample 'Custom Node' for users to copy paste
    """
    ids = 0  # used for pool of ids.

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

    # Add your own functions here.
