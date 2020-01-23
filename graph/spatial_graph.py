from itertools import count
from graph import BasicGraph


class Graph3D(BasicGraph):
    nid = count()
    """ a graph where all (x,y)-positions are unique. """
    def __init__(self, from_dict=None, from_list=None):
        super().__init__(from_dict=from_dict, from_list=from_list)

    def __copy__(self):
        g = Graph3D(from_dict=self.to_dict())
        return g

    @staticmethod
    def _check_tuples(n1):
        if not isinstance(n1, tuple):
            raise TypeError(f"expected tuple, not {type(n1)}")
        if len(n1) != 3:
            raise ValueError(f"expected tuple in the form as (x,y,z), got {n1}")
        if not all(isinstance(i, (float, int)) for i in n1):
            raise TypeError(f"expected all values to be integer or float, but got {n1}")

    @staticmethod
    def distance(n1, n2):
        """ returns the distance between to xyz tuples coordinates
        :param n1: (x,y,z)
        :param n2: (x,y,z)
        :return: float
        """
        Graph3D._check_tuples(n1)
        Graph3D._check_tuples(n2)
        (x1, y1, z1), (x2, y2, z2) = n1, n2
        a = abs(x2-x1)
        b = abs(y2-y1)
        c = abs(z2-z1)
        return (a * a + b + b + c + c) ** (1 / 2)

    def add_edge(self, n1, n2, value=None, bidirectional=False):
        self._check_tuples(n1)
        self._check_tuples(n2)

        super().add_edge(n1, n2, value, bidirectional)

    def add_node(self, node_id, obj=None):
        self._check_tuples(node_id)
        super().add_node(node_id, obj)
        """
        :param node_id: any hashable node.
        :param obj: any object that the node should refer to.

        PRO TIP: To retrieve the node obj use g.node(node_id)

        """
        self._nodes[node_id] = obj

    def n_nearest_neighbours(self, node_id, n=1):
        """ returns the node id of the `n` nearest neighbours. """
        self._check_tuples(node_id)
        if not isinstance(n, int):
            raise TypeError(f"expected n to be integer, not {type(n)}")
        if n < 1:
            raise ValueError(f"expected n >= 1, not {n}")

        d = [(self.distance(n1=node_id, n2=n), n) for n in self.nodes() if n != node_id]
        d.sort()
        if d:
            return [b for a, b in d][1:n+1]
        return None

    def plot(self, nodes=True, edges=True):
        """ plots nodes and links using matplotlib3"""
        from mpl_toolkits.mplot3d import Axes3D  # import required by matplotlib.
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Data for a three-dimensional line
        if edges:

            for edge in self.edges():
                n1, n2, v = edge
                xe, ye, ze = [], [], []
                xe.extend([n1[0], n2[0]])
                ye.extend([n1[1], n2[1]])
                ze.extend([n1[2], n2[2]])
                ax.plot3D(xe, ye, ze, 'gray')

        # Data for three-dimensional scattered points
        if nodes:
            xn, yn, zn, ix = [], [], [], []
            for idx, node in enumerate(self.nodes()):
                x, y, z = node
                xn.append(x)
                yn.append(y)
                zn.append(z)
                ix.append(idx)
            ax.scatter3D(xn, yn, zn, c=ix, cmap='Greens')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

