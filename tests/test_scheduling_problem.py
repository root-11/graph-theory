from graph.scheduling_problem import *


def test_01():
    """ Test the basics. """
    rdn = ResourceDemandNetwork()
    try:
        rdn.schedule()
        assert False, "No resources loaded yet."
    except SchedulingError:
        assert True

    m1 = Resource()
    process1 = Process(inputs=['a'], outputs=['A'])
    m1.add_process(process1)

    m2 = Resource()
    process2 = Process(outputs=['a'])  # no inputs required.
    m2.add_process(process2)

    rdn.add_edge(m1, m2)

    # The network is now ready to supply n 'A's when scheduled.
    m1.add_task(Task('a'))

    rdn.schedule()
    # Let's review one schedule...
    assert m2.tasks[0].name == 'a'
    assert m1.tasks[0].name == 'A'


def test_02():
    """ Test from the doctorate thesis with strict sequencing """

    # 1. Set the problem up:
    rdn = ResourceDemandNetwork()

    m1 = Resource()
    for o, i, t in zip('ABCDEFG', 'abcdefg', [14, 7, 3, 10, 5, 6, 6]):
        process = Process(inputs=[i], outputs=[o], run_time=t)
        m1.add_process(process)

    m2 = Resource()
    for o, t in zip('abcdefg', [14, 7, 3, 10, 5, 6, 6]):
        process = Process(outputs=[o], run_time=t)
        m2.add_process(process)
    rdn.add_edge(m1, m2)

    for task in 'ABCDEFG':
        m1.add_task(Task(task))

    # 2. solve the problem
    rdn.schedule()

    # 3. evaluate whether the output is correct.
    m1_expected = list('AEBDGFC')
    m1_jobs = [t.name for t in m1.tasks]
    assert m1_jobs == m1_expected

    m2_expected = list('AEBDGFC')
    m2_jobs = [t.name for t in m2.tasks]
    assert m2_jobs == m2_expected

    first_task = m1.tasks[0]
    assert isinstance(first_task, Task)
    assert first_task.scheduled_start == 2

    last_task = m1.tasks[-1]
    assert isinstance(last_task, Task)
    assert last_task.scheduled_finish == 53


def test_03():
    """ test for two independent demands for a single bottleneck """
    rdn = ResourceDemandNetwork()
    d1, d2, r = Resource(), Resource(), Resource()

    d1p = Process(inputs=['A'], outputs=['A'])
    d1.add_process(d1p)
    d2p = Process(inputs=['B'], outputs=['B'])
    d2.add_process(d2p)

    # p1 runtime < p2 runtime, so delivering p1 before p2 will minimise total lateness.
    p1 = Process(inputs=[], outputs=['A'], run_time=1)
    p2 = Process(inputs=[], outputs=['B'], run_time=2)
    r.add_process(p1)
    r.add_process(p2)

    rdn.add_edge(d1, r)
    rdn.add_edge(d2, r)

    d1.add_task(Task('A'))
    d2.add_task(Task('B'))

    rdn.schedule()

    # Check that we've minimised total lateness.
    assert r.tasks[0].name == 'A'
    assert r.tasks[1].name == 'B'


def test_04():
    """ test for two independent demands for a single bottleneck """
    rdn = ResourceDemandNetwork()

    d1, d2, r = Resource(), Resource(), Resource()

    # same run times == no deterministic solution.
    p1 = Process(inputs=[], outputs=['A'], run_time=1)
    p2 = Process(inputs=[], outputs=['B'], run_time=1)
    r.add_process(p1)
    r.add_process(p2)

    rdn.add_edge(d1, r)
    rdn.add_edge(d2, r)

    d1.add_task('A')
    d2.add_task('B')

    rdn.schedule()

    # Check that we've ordered tasks by id as the task have same runtime.
    assert r.tasks[0].name == 'A'
    assert r.tasks[1].name == 'B'


def test_dynamic_01():
    """ tests a network where tasks are added and removed between optimisation runs. """
    pass
    # 1. setup rdn
    # 2. add a few tasks
    # 3. schedule
    # 4. update a fictitious clock, as if some of the tasks where done.
    # 5. remove the tasks that would be done if the clock would progress.
    # (update the tasks with new start-times?)
    # 6. assert that the schedule doesn't change
    # 7. repeat from step 2.


