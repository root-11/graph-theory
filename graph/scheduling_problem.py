from graph import Graph
from itertools import count


class SchedulingError(ValueError):
    pass


class Task(object):
    _ids = count()

    def __init__(self, name):
        self.id = next(Task._ids)
        self.name = name
        # The values below are set by the Task.
        self.scheduled = False
        self.earliest_start = None
        self.latest_finish = None
        # The values below are set by the Resource
        self.earliest_finish = None
        self.latest_start = None
        self.duration = None
        self.cost = None

    def __str__(self):
        return f"Task {self.name}"

    def __hash__(self):
        return self.id

    def __bool__(self):
        return self.scheduled

    def copy(self):
        return Task(self.name)


class Resource(object):
    """
    A Resource can perform a single process at any time.
    """
    _ids = count()

    def __init__(self):
        self.id = next(Resource._ids)
        self._rdn = None
        self.tasks = []
        self.processes = []

    def __str__(self):
        return f"{self.__class__.__name__}({self.id})"

    def __hash__(self):
        return self.id

    @property
    def rdn(self):
        return self._rdn

    @rdn.setter
    def rdn(self, value):
        if not isinstance(value, ResourceDemandNetwork):
            raise ValueError("Node.rdn is a reserved property.")
        self._rdn = value

    def add_task(self, task):
        t = Task(task)
        assert isinstance(t, Task)
        self.tasks.append(task)

    def add_process(self, proc):
        assert isinstance(proc, Process)
        self.processes.append(proc)

    def can_support(self, task):
        for process in self.processes:
            assert isinstance(process, Process)
            if task.name in process.outputs:
                return True
        return False

    def suppliers(self):
        if not isinstance(self._rdn, ResourceDemandNetwork):
            raise AttributeError("Add Node to ResourceDemandNetwork first.")
        rdn = self._rdn.network
        assert isinstance(rdn, Graph)
        return [rdn.node(nid) for nid in rdn.nodes(from_node=self.id)]

    def schedule(self):
        """ find suppliers from self.rdn and assign tasks """
        undone = [t for t in self.tasks if not t]
        all_suppliers = set()
        for task in undone:
            supplier_short_list = set()

            for supplier in self.suppliers():
                assert isinstance(supplier, Resource)
                if supplier.can_support(task):
                    supplier_short_list.add(supplier)
                    supplier.add_task(task.copy())

            if not supplier_short_list:
                raise SchedulingError(f"No suppliers for {task} with {self}")
            else:
                all_suppliers.update(supplier_short_list)

        for s in all_suppliers:
            s.schedule()


class Process(object):
    def __init__(self,
                 inputs=None, outputs=None,
                 setup_time=0, run_time=0, shutdown_time=0, change_over_time=0,
                 cost=0):
        if inputs is not None:
            assert isinstance(inputs, list)
        self.inputs = inputs
        if outputs is not None:
            assert isinstance(outputs, list)
        self.outputs = outputs

        for i in [setup_time, run_time, shutdown_time, change_over_time, cost]:
            if not isinstance(i, (int, float)):
                raise TypeError(f"{i.__name__} is {type(i)}. Expected float or int")

        # The values below are set by the resource.
        self.setup_time = setup_time
        self.run_time = run_time
        self.shutdown_time = shutdown_time
        self.change_over_time = change_over_time
        self.cost = cost


class ResourceDemandNetwork(object):
    """ Resource-Demand Network
    Resources (machines, people, inventory) are connected in a network, where
    the edges permit transformation of tasks.

    All demands are generated at the sink.
    All resources are generated at the source.
    """
    def __init__(self):
        self.network = Graph()
        self.tasks = []
        self._schedule = {}  # task id, node id, task,
        self.sinks = set()
        self.sources = set()
        self.resources = set()

    def add_task(self, task):
        """ Creates task at the sink"""
        self.tasks[task.id] = task

    def add_resource(self, resource):
        """ adds resource to RDN. """
        assert isinstance(resource, Resource)
        self.network.add_node(node_id=resource.id, obj=resource)
        resource.rdn = self

    def add_edge(self, client, resource):
        """ A directed edge"""
        assert isinstance(client, Resource)
        assert isinstance(resource, Resource)
        self.network.add_node(node_id=client.id, obj=client)
        self.network.add_node(node_id=resource.id, obj=resource)
        self.network.add_edge(client.id, resource.id, value=1)  # hop size = 1.

    def schedule(self):
        self.sinks = {self.network.nodes(out_degree=0)}
        for task in self.tasks:
            # auction!  # TODO:FIXME!
            # any resource that can deliver the task, should bid.
            pass
        self.sources = {self.network.nodes(in_degree=0)}
        while any(not task.complete for task in self.tasks):
            _ = [r.schedule() for r in self.resources]

