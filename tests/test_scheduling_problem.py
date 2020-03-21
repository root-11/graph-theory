from graph.scheduling_problem import *


def test_01():
    """ Test from the doctorate thesis"""
    rdn = ResourceDemandNetwork()
    rdn.add_task('A')
    try:
        rdn.schedule()
        assert False, "No resources loaded yet."
    except SchedulingError:
        assert True
    m1 = Resource()
    process1 = Process(inputs=['a'], outputs=['A'])
    m1.add_process(process1)

    try:
        rdn.schedule()
        assert False
    except SchedulingError:
        pass

    m2 = Resource()
    process2 = Process(outputs=['a'])
    m2.add_process(process2)
    rdn.edge(m1, m2)
    # The network is now ready to supply n 'A's using rdn.add_task('A')
    # Let's review one schedule...
    rdn.schedule()


def test_02():
    """ Test from the doctorate thesis with strict sequencing """
    pass
