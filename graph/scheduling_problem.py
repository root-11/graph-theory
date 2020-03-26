from graph import Graph
from itertools import count


class SchedulingError(ValueError):
    pass


class Task(object):
    _ids = count()

    def __init__(self, requires, earliest_start=None, latest_finish=None):
        """
        :param requires: multiple types:
            dicts are taken as given.
            sets, lists, tuples are transformed to dicts with key: count(key)
            strings are handled as {str: 1}
        :param earliest_start: sortable value
        :param latest_finish: sortable value
        """
        if isinstance(requires, (set, tuple, list)):
            self.requires = {k: requires.count(k) for k in set(requires)}
        elif isinstance(requires, str):
            self.requires = {requires: 1}
        elif isinstance(requires, dict):
            self.requires = requires
        else:
            raise TypeError(f"requires expected dict, tuple, list, set or str, but got {type(requires)}")

        self.earliest_start = earliest_start
        self.latest_finish = latest_finish
        # The values below are set by the Resource
        self.earliest_finish = None
        self.latest_start = None
        self.duration = None
        self.cost = None

        self.scheduled_start = None
        self.scheduled_finish = None

        self.id = next(self._ids)

    def __eq__(self, other):
        """ returns True if process.output.keys() == task.requires.keys()"""
        if isinstance(other, Process):
            return self.requires.keys() == other.outputs.keys()

        raise NotImplementedError  # ... if we haven't returned above ...

    def __bool__(self):
        return self.scheduled_start is not None and self.scheduled_finish is not None

    def __str__(self):
        return f"{self.__class__.__name__}({self.id}) {self.requires}"

    def __hash__(self):
        return self.id

    def copy(self):
        return Task(self.requires)


class Resource(object):
    """
    A Resource can perform a single process at any time.
    """
    _ids = count()

    def __init__(self):
        self.id = next(self._ids)
        self.mq = []
        self.tasks = []
        self._task_ids = set()
        self.processes = []
        self._rdn = None

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
            err = f"{self.__class__.__name__}.rdn can only be set by the {ResourceDemandNetwork.__name__}"
            raise ValueError(err)
        if self._rdn is not None:
            raise ValueError(f"{self}.rdn has already been set.")
        self._rdn = value

    def add_task(self, task):
        assert isinstance(task, Task)
        if task.id in self._task_ids:
            raise ValueError(f"{task} is already registered with {self}")
        else:
            self._task_ids.add(task.id)
        self.tasks.append(task)
        self.mq.append(task.id)

    def remove_task(self, task):
        assert isinstance(task, Task)
        self._task_ids.discard(task.id)
        if task in self.tasks:
            self.tasks.remove(task)
        if task in self.mq:
            self.mq.append(task.id)

    def add_process(self, proc):
        assert isinstance(proc, Process)
        self.processes.append(proc)

    def can_support(self, task):
        for process in self.processes:
            assert isinstance(process, Process)
            if process == task:
                return True
        return False

    def schedule(self):
        """ calculates the schedule. Stores the schedule in self.tasks """
        if not isinstance(self._rdn, ResourceDemandNetwork):
            raise AttributeError(f"Use ResourceDemandNetwork.add_resource({self}) first.")

        # discover tasks
        # for each task:
        #   find supply requirements.
        #   find suppliers
        #   for each supplier:
        #       create task
        #       add task to supplier

        # if all supply tasks are scheduled for delivery:
        # for each task:
        #   put own tasks in sequence

        # sort to schedule.
        # detect idletime, lateness, ...
        # remove any duplicate tasks (and ping suppliers to update delivery times).

        # notify customer of change in delivery times.

        suppliers = [r_id for r_id in self._rdn.network.nodes(from_node=self.id)]

        undone = [t for t in self.tasks if not t]
        all_suppliers = set()
        for task in undone:
            supplier_short_list = set()

            for supplier in self.suppliers():
                assert isinstance(supplier, Resource)
                if supplier.can_support(task):
                    supplier_short_list.add(supplier)
                    supplier.add_task(task.copy())  # <--- this isn't right.

            if not supplier_short_list:
                raise SchedulingError(f"No suppliers for {task} with {self}")
            else:
                all_suppliers.update(supplier_short_list)

        for s in all_suppliers:
            s.schedule()


class Process(object):
    def __init__(self, inputs=None, outputs=None,
                 setup_time=0, run_time=0, shutdown_time=0, change_over_time=0, cost=0):
        """
        :param inputs: multiple types:
            dicts are taken as given.
            sets, lists, tuples are transformed to dicts with key: count(key)
            strings are handled as {str: 1}
        :param outputs: multiple types:
            dicts are taken as given.
            sets, lists, tuples are transformed to dicts with key: count(key)
            strings are handled as {str: 1}
        :param setup_time: any sortable value
        :param run_time: any sortable value
        :param shutdown_time: any sortable value
        :param change_over_time: any sortable value
        :param cost: any sortable value
        """
        if inputs is None:
            self.inputs = {}
        elif isinstance(inputs, (set, tuple, list)):
            self.inputs = {k: inputs.count(k) for k in set(inputs)}
        elif isinstance(inputs, str):
            self.inputs = {inputs: 1}
        elif isinstance(inputs, dict):
            self.inputs = inputs
        else:
            raise TypeError(f"inputs expected set, tuple, list or dict, but got {type(inputs)}")

        if outputs is None:
            self.outputs = {}
        elif isinstance(outputs, (set, tuple, list)):
            self.outputs = {k: outputs.count(k) for k in set(outputs)}
        elif isinstance(outputs, str):
            self.outputs = {outputs: 1}
        elif isinstance(outputs, dict):
            self.outputs = outputs
        else:
            raise ValueError(f"outputs expected set, tuple, list or dict, but got {type(outputs)}")

        for i in [setup_time, run_time, shutdown_time, change_over_time, cost]:
            if not isinstance(i, (int, float)):
                raise TypeError(f"{i.__name__} is {type(i)}. Expected float or int")

        # The values below are set by the resource.
        self.setup_time = setup_time
        self.run_time = run_time
        self.shutdown_time = shutdown_time
        self.change_over_time = change_over_time
        self.cost = cost

    def __eq__(self, other):
        """ return True if process.output.keys() == task.requires.keys() 
        Note:
        As one 'production' could supply multiple tasks, the challenge is to
        match tasks with productions, but should probably be performed in 
        another function.
        """
        if isinstance(other, Task):
            return self.outputs.keys() == other.requires.keys()

        raise NotImplementedError  # ... if we haven't returned above ...


class ResourceDemandNetwork(object):
    """ Resource-Demand Network
    Resources (machines, people, inventory) are connected in a network, where
    the edges permit transformation of tasks.

    Tasks must be given directly to Resources
    """
    def __init__(self):
        self.network = Graph()
        self.mq = []

    def add_resource(self, resource):
        """ adds resource to RDN. """
        assert isinstance(resource, Resource)
        if resource.id in self.network.nodes():
            raise ValueError(f"Duplicate entry: {resource} is already registered.")

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
        # 1. discover new tasks:
        resources = self.network.nodes()
        if not resources:
            raise SchedulingError("No resources to schedule.")

        for r in resources:
            if r.mq:
                self.mq.append(r.id)

        # 2. initiate the scheduling.
        while self.mq:
            r_id = self.mq.pop(0)  # resource id.
            resource = self.network.node(node_id=r_id)
            assert isinstance(resource, Resource)
            resource.schedule()

