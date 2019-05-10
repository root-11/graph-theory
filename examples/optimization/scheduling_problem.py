from graph import Graph

__description__ = """

solve a scheduling problem as a graph with custom nodes.

"""


class Machine(object):
    def __init__(self, processes=None):
        self.processes = processes
        self._bom = Graph()
        self._cot = Graph()
        self._jobs = Graph()
        self._supplies = Graph()
        self._schedule = Graph()

    def bill_of_materials(self, bom):
        assert isinstance(bom, Graph)
        self._bom = bom

    def change_over_times(self, cot):
        assert isinstance(cot, Graph)
        self._cot = cot

    def jobs(self, jobs):
        """
        Receives schedule (graph)
        :param jobs: class Graph.
        :return: None

        Expects jobs as a graph with individual jobs as subgraphs in the graph..
        """
        assert isinstance(jobs, Graph)
        self._jobs = jobs

    def supply(self, schedule):
        """
        Receives supply schedule (graph)
        :param schedule: class Graph
        :return: None

        Expects supply schedule as a bipartite graph with job delivery times
        (left) partition and any delay (right partition)
        """
        assert isinstance(schedule, Graph)
        self._schedule = schedule

    def schedule(self):
        """
        Calculates a schedule for the Machine.
        :return: Suggested changes.
        """
        orders, supplies = 0, 0
        return orders, supplies


def schedule(machines):
    """
    :param machines: Graph with machines and their links to other machines.
    :param bom: Graph (bill of materials).
    :param co: Graph with change over times.
    :param jobs: Graph with orders and items in the order.
    :return: Graph.
    """
    assert isinstance(machines, Graph)
    for _id in machines.nodes():
        m = machines[_id]
        assert isinstance(m, Machine)
        order, supplies = m.schedule()
    # TODO
    # read the jobs.
    # determine supplies
    # calculate the minimum change over sequence.
    # send request for supplies
    return Graph()


def single_machine_scheduling_problem():
    """
    single item order that requires assembly:

        item-1 ---+--> order-1
        item-2 ---|

        item-1 ---+--> order-2
        item-2 ---|

    To switch from product of item 1 to item 2 there is a change-over time.
    """
    m1 = Machine()
    m1.bill_of_materials(Graph(from_list=[
        ("raw-1", "item-1", 3),
        ("raw-2", "item-1", 3),
        ("raw-3", "item-2", 4),
        ("raw-4", "item-2", 2),
    ]))
    m1.change_over_times(Graph(from_list=[
        ("item-1", "item-1", 0),  # repeat production.
        ("item-2", "item-2", 0),
        ("item-1", "item-2", 1),  # change over cost from item 1 to item 2.
        ("item-2", "item-1", 2),
    ]))
    m1.jobs(Graph(from_list=[
        ("item-1", "order-1", 0),  # order 1 for item 1.delivered at time zero
        ("item-2", "order-1", 0),
        ("item-1", "order-2", 0),
        ("item-2", "order-2", 0),
    ]))
    m1.supply(Graph(from_list=[
        (0, "raw-1", 0),
        (0, "raw-2", 0),
        (0, "raw-3", 0),
        (0, "raw-4", 0),
    ]))

    machines = Graph()
    machines.add_node(node_id=1, obj=m1)
    s = schedule(machines)
    assert isinstance(s, Graph)


def two_machine_scheduling_problem():
    """
    supplies --> M1 --> M2 <--- orders.
    """
    # configure the machines:
    m1 = Machine()
    m1.bill_of_materials(Graph(from_list=[
        ("raw-1", "item-1", 3),
        ("raw-1", "item-1", 3),
        ("raw-1", "item-2", 4),
        ("raw-1", "item-2", 2),
    ]))
    m1.change_over_times(Graph(from_list=[
        ("item-1", "item-1", 0),  # repeat production.
        ("item-2", "item-2", 0),
        ("item-1", "item-2", 1),  # change over cost from item 1 to item 2.
        ("item-2", "item-1", 2),
    ]))
    m1.supply(Graph(from_list=[
        (0, "raw-1", 0),
        (0, "raw-1", 1),
        (0, "raw-1", 2),
        (0, "raw-1", 3)
    ]))

    m2 = Machine()
    m2.bill_of_materials(Graph(from_list=[
        # raw  ---> product, time
        ("item-1", "product-1", 4),
        ("item-2", "product-1", 4),
        ("item-1", "product-2", 4),
        ("item-1", "product-2", 4),
    ]))
    m2.change_over_times(Graph(from_list=[
        ("product-1", "product-1", 1),
        ("product-1", "product-2", 2),
        ("product-2", "product-2", 1),
        ("product-2", "product-1", 2),
    ]))
    m2.jobs(Graph(from_list=[
        ("product-1", "order-1", 0),  # order 1 for product 1.delivered at time zero
        ("product-2", "order-1", 0),
        ("product-1", "order-2", 0),
        ("product-2", "order-2", 0),
    ]))

    machines = Graph()
    machines.add_node(node_id=1, obj=m1)
    machines.add_node(node_id=2, obj=m2)
    machines.add_edge(node1=1, node2=2, value=0)

    s = schedule(machines)
    assert isinstance(s, Graph)


def three_machine_scheduling_problem_shared_resource():
    """
    R1 --\
         +--- R3
    R2 --/
    """
    pass


def three_machine_scheduling_problem_bottleneck():
    """
          /-- R2
    R1 --+
          \-- R3
    """
    pass


def four_machine_scheduling_problem():
    """
    4 robot scheduling system

         /--- R2 ---\
    R1 --+          +-- R4.
         \--- R3 ---/
    """
    pass



