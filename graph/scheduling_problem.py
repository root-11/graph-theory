from graph import Graph
from itertools import count


class SchedulingError(ValueError):
    pass


class NullTask(object):
    scheduled_finish = 0


class Task(object):
    _ids = count()

    def __init__(self, requires, client=None, supplier=None, earliest_start=None, latest_finish=None,
                 latest_start=None, earliest_finish=None, duration=None, cost=None):
        """
        :param requires: multiple types:
            dicts are taken as given.
            sets, lists, tuples are transformed to dicts with key: count(key)
            strings are handled as {str: 1}
        :param client: the resource requiring the task to be completed.
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

        if client is None or isinstance(client, int):
            self.client = client
        else:
            raise TypeError(f"expected client to be int, not {type(client)}")

        if supplier is None or isinstance(supplier, int):
            self.supplier = supplier
        else:
            raise TypeError(f"expected supplier to be int, not {type(client)}")

        self.earliest_start = earliest_start
        self.earliest_finish = earliest_finish
        self.latest_start = latest_start
        self.latest_finish = latest_finish
        self.duration = duration
        self.cost = cost

        self.scheduled_start = None
        self.scheduled_finish = None

        self.id = next(self._ids)

    def __eq__(self, other):
        """ returns True if process.output.keys() == task.requires.keys()"""
        if isinstance(other, Process):
            return self.requires.keys() == other.outputs.keys()
        elif isinstance(other, Task):
            return self.requires.keys() == other.requires.keys()
        else:
            raise NotImplementedError  # ... if we haven't returned above ...

    def __bool__(self):
        return self.scheduled_start is not None and self.scheduled_finish is not None

    def __str__(self):
        return f"{self.__class__.__name__}({self.id}) {self.requires}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self.id

    def copy(self):
        return Task(requires=self.requires,
                    client=self.client,
                    earliest_start=self.earliest_start,
                    latest_finish=self.latest_finish,
                    latest_start=self.latest_start,
                    earliest_finish=self.earliest_finish,
                    duration=self.duration,
                    cost=self.cost)


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

    def __str__(self):
        return f"{self.__class__.__name__}: {self.outputs} <- {self.inputs}"

    def __repr__(self):
        return str(self)

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


class Resource(object):
    """
    A Resource can perform a single process at any time.
    """
    _ids = count()

    def __init__(self):
        self.id = next(self._ids)
        self.new_tasks = []  # new tasks
        self.tasks = []  # known tasks
        self.supply = {}  # task = [list of tasks given to suppliers]
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
        if self._rdn is None:
            raise ValueError(f"{self} has not been added to the RDN. Do this first.")
        if not any(p == task for p in self.processes):
            raise ValueError(f"{self} has no process supporting {task}")
        self.new_tasks.append(task)
        self._rdn.notify(r_id=self.id)

    def remove_task(self, task):
        assert isinstance(task, Task)
        if task in self.tasks:
            self.tasks.remove(task)
        if task in self.new_tasks:
            self.new_tasks.remove(task)

    def add_process(self, proc):
        assert isinstance(proc, Process)
        self.processes.append(proc)

    def supplies(self, task):
        for p in self.processes:
            if p == task:
                return p.inputs
        raise ValueError(f"{self} does not support {task}")

    def get_process(self, task):
        for p in self.processes:
            if p == task:
                return p
        return None

    def can_support(self, task):
        """ returns bool; True if there's a process that can support the task """
        if any(p == task for p in self.processes):
            return True
        return False

    def notify(self):
        """ api for other resources to ping, so that this resource is updated.

        when customers add tasks to resources, they use resource.add_task(Task).
        This method puts the id of the resource onto the RDNs message queue, so that
        the RDN will identify the resource and run its .schedule() method.

        The opposite direction is more indirect, as a resource may finish a schedule,
        but has no way of notifying the customer. To overcome this, the resource can
        identify the customer from the Task.client, and `notify` the client using this
        method.
        """
        if self._rdn is None:
            raise ValueError(f"{self} has not been added to the RDN. Do this first.")
        self._rdn.notify(r_id=self.id)

    def suppliers(self, task):
        """ returns list of suppliers who can support the task. """
        assert isinstance(task, Task)
        suppliers = []
        for nid in self._rdn.network.nodes(from_node=self.id):
            resource = self._rdn.network.node(nid)
            assert isinstance(resource, Resource)
            if resource.can_support(task):
                suppliers.append(resource)
        return suppliers

    def schedule(self):
        """ calculates the schedule. Stores the schedule in self.tasks

        1. discover tasks:
            tasks are in self.mq
            for each new task:
                find supply requirements (proc.output requires proc.input)
                find suppliers
                for each supplier:
                    create a new (supply) task as: Task(requires=supply, required_by=self.id)
                    add task to supplier (which automatically notifies)
                    notify via supplier.notify()

        2. determine better schedule (if all supply tasks are scheduled for delivery):
        sort tasks by arrival (as we hold no inventory)

        3. detect idle-time, lateness, ...
        remove any duplicate tasks (and ping suppliers to update delivery times).

        notify client of change in delivery times.

        """
        if not isinstance(self._rdn, ResourceDemandNetwork):
            raise AttributeError(f"Use ResourceDemandNetwork.add_resource({self}) first.")

        # 1. discover tasks
        while self.new_tasks:
            task = self.new_tasks.pop(0)
            assert isinstance(task, Task), type(task)
            self.tasks.append(task)
            process = self.get_process(task)
            assert isinstance(process, Process)
            if not process.inputs:
                continue  # nothing to do. supplies are not required.

            self.supply[task] = []  # create empty list for adding tasks given to suppliers.
            # below:
            # if there are more customers to a supplier, it'll be ambiguous, who's ordering what.
            # Therefore we set: Task.client = self.id.
            supply_task = Task(requires=process.inputs, client=self.id)

            for resource in self.suppliers(supply_task):
                new_task = Task(requires=process.inputs, client=self.id, supplier=resource.id)
                self.supply[task].append(new_task)  # own reference to task.
                resource.add_task(new_task)  # adding task suppliers inbox.
            return  # as this resource created tasks, it will have to wait for reply from suppliers

        for supply_task, supplier_list in self.supply.items():
            if any(not t for t in supplier_list):
                return  # resource is waiting for information from suppliers.

        # as self.supply contains all scheduled supply tasks, it is possible to iterate through own
        # tasks and determine the best supply option without having to find the supplier.
        # -- some kind of match(self.tasks, self.supply)

        # PS> remember to cancel duplicate tasks for stuff that isn't needed.

        # 2. check / determine schedule.

        # to minimise total idle time, sort tasks by (runtime, name)
        assert all(isinstance(t, Task) for t in self.tasks)  # this is disabled with python -OO
        tasks = [(t.runtime, t.name, t) for t in self.tasks]
        # `runtime` to minimise initial waiting time, and,
        # `name` to assure that task of same name come out together so c/o time can be exploited.
        tasks.sort()  # ascending!
        self.tasks = [t for r, n, t in tasks]

        # for task in tasks: update start and finish time.
        previous_task = NullTask
        for ix, task in enumerate(self.tasks):
            process = self.get_process(task)
            assert isinstance(process, Process)
            assert isinstance(task, Task)

            if previous_task == task:  # handling change_over time instead of shutdown time if the tasks are the same.
                previous_task.scheduled_finish += (process.change_over_time - process.shutdown_time)

            if process.inputs:  # pick best matching supply time.
                supply_times = [(t.scheduled_finish, t) for t in self.supply[task]]
                supply_times.sort()
                supply_times = [t for _, t in supply_times]

                fas = supply_times[0]  # pick first available supply
                for t in supply_times[1:]:  # cancel all other tasks
                    resource = self._rdn.network.node(node_id=t.supplier)
                    assert isinstance(resource, Resource)
                    resource.remove_task(t)
                    resource.notify()

                start_time = max(previous_task.scheduled_finish, fas.scheduled_finish)
            else:
                start_time = previous_task.scheduled_finish

            task.scheduled_start = start_time

            if previous_task == task:
                steps = [task.scheduled_start, process.run_time, process.shutdown_time]
            else:
                steps = [task.scheduled_start, process.setup_time, process.run_time, process.shutdown_time]
            task.scheduled_finish = sum(steps)

            # get ready to loop.
            previous_task = task

        # at this point, the tasks are all scheduled with start and finish times.

        # check if there's any idle time.
        # notify customer.

        print(f"{self} done")




class ResourceDemandNetwork(object):
    """ Resource-Demand Network
    Resources (machines, people, inventory) are connected in a network, where
    the edges permit transformation of tasks.

    Tasks must be given directly to Resources
    """
    def __init__(self):
        self.network = Graph()
        self.task_queue = {}  # only keys are used as this behaves like an ordered set

    def add_resource(self, resource):
        """ adds resource to RDN. """
        assert isinstance(resource, Resource)
        if resource.id not in self.network.nodes():
            self.network.add_node(node_id=resource.id, obj=resource)
            resource.rdn = self
        # else: ... just return.

    def add_edge(self, client, resource):
        """ A directed edge"""
        assert isinstance(client, Resource)
        assert isinstance(resource, Resource)
        self.network.add_node(node_id=client.id, obj=client)
        self.network.add_node(node_id=resource.id, obj=resource)
        self.network.add_edge(client.id, resource.id, value=1)  # hop size = 1.
        if client.rdn is None:
            client.rdn = self
        if resource.rdn is None:
            resource.rdn = self

    def notify(self, r_id):
        """ method for assure that RDN will run resource.schedule() """
        if r_id not in self.task_queue:
            self.task_queue[r_id] = self.network.node(r_id)
        # else: ... just return.

    def schedule(self):
        """ schedules the tasks on the resources """
        # 1. discover new tasks and prepare the task_queue:
        resource_ids = self.network.nodes()
        if not resource_ids:
            raise SchedulingError("No resources to schedule.")

        for r_id in resource_ids:
            resource = self.network.node(r_id)
            if resource.mq:
                self.task_queue[r_id] = resource
        # task_queue is now prepared.

        # 2. initiate the scheduling.
        while self.task_queue:
            # first swop the pointer for the queue, so I can loop over it, without
            # changing the order of events.
            current_queue, self.task_queue = self.task_queue, {}

            for r_id, resource in current_queue.items():
                resource = self.network.node(node_id=r_id)
                assert isinstance(resource, Resource)
                resource.schedule()
                # self.task_queue is now populated by:
                # - any resource that places tasks on any downstream resource
                # - any resource that finishes scheduling and uses notify


