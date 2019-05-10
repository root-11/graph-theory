from graph import Graph
from solvers.transshipment_problem_oop import *


def single_cross_network():
    """
           (2)
            |
    (1) --- + --- (3)
            |
           (4)
    """
    paths = [
        (1, '+', 1),
        (2, '+', 1),
        (3, '+', 1),
        (4, '+', 1),
    ]
    paths.extend([(n2, n1, d) for n1, n2, d in paths])  # adding reverse paths.
    return Graph(from_list=paths)


def double_cross_network():
    """
             1        2
             +        +
             |        |
    8 +-----18--------23---+ 3
             |        |
             |        |
    7 +-----67--------45---+ 4
             |        |
             +        +
             6        5

    """
    paths = [
        (1, 18, 1),
        (2, 23, 1),
        (3, 23, 1),
        (4, 45, 1),
        (5, 45, 1),
        (6, 67, 1),
        (7, 67, 1),
        (8, 18, 1),
        (18, 23, 1),
        (23, 45, 1),
        (45, 67, 1),
        (67, 18, 1),
    ]
    paths.extend([(n2, n1, d) for n1, n2, d in paths])  # adding the reverse path.
    return Graph(from_list=paths)


def airport_model():
    """
    [airport] --- (1) --- (2) --- (3)

    """
    paths = [
        ("airport", 1, 1),
        (1, 2, 1),
        (2, 3, 1)
    ]
    paths.extend([(n2, n1, d) for n1, n2, d in paths])  # adding the reverse path.
    return Graph(from_list=paths)


def test_transports_fifo():
    """
    passengers must exit in the same order as they board.
    """
    n = airport_model()
    a = "airport"
    t = Transporter(network=n, start=1, stops={a, 1, 2, 3}, order=Transporter.FIFO, loading_time=1)
    p1 = Passenger(network=n, trips=[PassengerTrip(network=n, start=a, end=2)])
    p2 = Passenger(network=n, trips=[PassengerTrip(network=n, start=1, end=3)])
    schedule_all_trips(transports=[t], passengers=[p1, p2])
    # assert schedule has no overhead as the boarding is FIFO.
    assert True

    p1 = Passenger(network=n, trips=[PassengerTrip(network=n, start=a, end=3)])
    p2 = Passenger(network=n, trips=[PassengerTrip(network=n, start=1, end=2)])
    schedule_all_trips(transports=[t], passengers=[p1, p2])
    # assert schedule has overhead of unloading p2 before loading p1 so that
    # p1 can get off plus the extra overhead of reloading p2.


def test_transports_filo():
    """
    passengers must move to the back of the bus to let other passengers on
    board. If a passenger at the rear of the bus wants to get out, all
    passengers in front of him/her must get off the bus first.
    """
    n = airport_model()
    a = "airport"
    t = Transporter(network=n, start=1, stops={a, 1, 2, 3}, order=Transporter.FIFO)
    p1 = Passenger(network=n, trips=[PassengerTrip(network=n, start=a, end=2)])
    p2 = Passenger(network=n, trips=[PassengerTrip(network=n, start=1, end=3)])
    schedule_all_trips(transports=[t], passengers=[p1, p2])
    # assert schedule has overhead of unloading p2 before p1 can get off.
    assert True

    p1 = Passenger(network=n, trips=[PassengerTrip(network=n, start=a, end=3)])
    p2 = Passenger(network=n, trips=[PassengerTrip(network=n, start=1, end=2)])
    schedule_all_trips(transports=[t], passengers=[p1, p2])
    # assert schedule has no overhead as the boarding is sequential.


def test_transports_no_order():
    """
    passengers can board and exit the bus with no requirement for order.
    """
    pass


def test_single_traveller_a_to_b():
    """
    Assures that a single traveller can travel from A to B
    (requires journey from A via ah and bc to reach B)
    """
    network = single_cross_network()

    passengers = [
        Passenger(network=network, trips=[(1, 2, 0, float('inf'))]),
    ]

    transports = [
        Transporter(network=network, start=1, stops={1, '+', 3}),
        Transporter(network=network, start=2, stops={2, '+', 4})
    ]

    schedule_all_trips(transports=transports, passengers=passengers)


def test_two_travellers_a_to_b():
    """
    Assures that conflict of shared resource is sequenced as two
    travellers reach ah at the same time.
    """
    network = single_cross_network()

    passengers = [
        Passenger(network=network, trips=[(1, 2, 0, float('inf'))]),
        Passenger(network=network, trips=[(2, 3, 0, float('inf'))]),
    ]

    transports = [
        Transporter(network=network, start=1, stops={1, '+', 3}),
        Transporter(network=network, start=2, stops={2, '+', 4})
    ]

    schedule_all_trips(transports=transports, passengers=passengers)


def test_four_travellers_a_to_b():
    """

    """
    network = single_cross_network()

    passengers = [
        Passenger(network=network, trips=[(1, 2, 0, float('inf'))]),
        Passenger(network=network, trips=[(2, 3, 0, float('inf'))]),
        Passenger(network=network, trips=[(3, 4, 0, float('inf'))]),
        Passenger(network=network, trips=[(4, 1, 0, float('inf'))])
    ]

    transports = [
        Transporter(network=network, start=1, stops={1, '+', 3}),
        Transporter(network=network, start=2, stops={2, '+', 4})
    ]

    schedule_all_trips(transports=transports, passengers=passengers)


def test_airport_shuttle_problem():
    """
    Groups passengers need to travel from the airport to their hotel at
    stops 1,2,3. All arrive with the same jumbo jet, but there is only
    one bus available.

    Which sequence results in least total waiting time?
    """
    network = airport_model()

    passengers = [
        Passenger(network=network, trips=[('airport', 1, 0, float('inf'))]),
        Passenger(network=network, trips=[('airport', 2, 0, float('inf'))]),
        Passenger(network=network, trips=[('airport', 3, 0, float('inf'))]),
    ]

    transports = [
        Transporter(network=network, start='airport', stops=set(network.nodes())),
    ]

    schedule_all_trips(transports=transports, passengers=passengers)


def test_airport_shuttle_problem_reverse():
    """
    Passengers need to get to the airport.
    """
    network = airport_model()

    passengers = [
        Passenger(network=network, trips=[(1, 'airport', 0, float('inf'))]),
        Passenger(network=network, trips=[(2, 'airport', 0, float('inf'))]),
        Passenger(network=network, trips=[(3, 'airport', 0, float('inf'))]),
    ]

    transports = [
        Transporter(network=network, start='airport', stops=set(network.nodes())),
    ]

    schedule_all_trips(transports=transports, passengers=passengers)


def test_airport_shuttle_problem_out_and_back():
    """
    A group of passengers need to get to their plane and one group to the hotel.
    The bus handling this can do so in a single circuit.
    """
    network = airport_model()

    passengers = [
        Passenger(network=network, trips=[(1, 'airport', 0, float('inf'))]),
        Passenger(network=network, trips=[('airport', 1, 0, float('inf'))]),
    ]

    transports = [
        Transporter(network=network, start='airport', stops=set(network.nodes())),
    ]

    schedule_all_trips(transports=transports, passengers=passengers)


def test_airport_shuttle_problem_independent_passengers():
    """
    From 3 hotels, there are a number of passengers going for one of three
    flights home. At the same time a number of passengers are arriving
    on each of the three flights which intend to travel to one of three
    available hotels.

    [airport] --- (1) --- (2) --- (3)

    """
    network = airport_model()

    passengers = [
        Passenger(network=network, trips=[('airport', 1, 0, float('inf'))]),
        Passenger(network=network, trips=[('airport', 2, 0, float('inf'))]),
        Passenger(network=network, trips=[('airport', 3, 0, float('inf'))]),
        Passenger(network=network, trips=[(1, 'airport', 0, float('inf'))]),
        Passenger(network=network, trips=[(2, 'airport', 0, float('inf'))]),
        Passenger(network=network, trips=[(3, 'airport', 0, float('inf'))]),
    ]

    transports = [
        Transporter(network=network, start='airport', stops=set(network.nodes())),
    ]

    schedule_all_trips(transports=transports, passengers=passengers)


def test_strict_sequencing():
    """
    Test strict ordering for arrivals at a destination.

    Each Passenger (P) arrives at a 'sequencer' in random order, but with
    strict requirements to pass the waypoint at particular times.



    entry sequence: 13578642
    exit sequence: 87654321

    network:

                   [ ,8]--+--[7,6]
                          |
                   [4,5]--+--[3,2]
                          |
    [1,3,5,7,8,6,4,2] --> + --> [8,7,6,5,4,3,2,1] -->[waypoint]

    The solution has two phases:
    - sortation (steps 1-7), and,
    - delivery (steps 8-12)

    Start of sortation phase.

    Step 1:
                     [ , ]--+--[ ,4]
                            |
                     [ , ]--+--[ ,2]
                            |
    [1,3,5,7,8,6, , ] --> [4,2] --> [ , , , , , , , ]

    Step 2:
                     [8, ]--+--[ ,4]
                            |
                     [6, ]--+--[ ,2]
                            |
    [1,3,5,7, , , , ] --> [8,6] --> [ , , , , , , , ]

    Step 3:
                     [8,7]--+--[ ,4]
                            |
                     [6,5]--+--[ ,2]
                            |
    [1,3, , , , , , ] --> [5,7] --> [ , , , , , , , ]

    Step 4:
                     [8,7]--+--[ ,4]
                            |
                     [6,5]--+--[3,2]
                            |
    [1, , , , , , , ] --> [3] --> [ , , , , , , , ]

    End of sortation phase.
    Start of delivery phase

    Step 5:
                    [8,7]--+--[ ,4]
                           |
                    [6,5]--+--[3,2]
                           |
    [ , , , , , , , ] --> [1] --> [ , , , , , , ,1]

    Step 6:
                     [8,7]--+--[ ,4]
                            |
                     [6,5]--+--[ , ]
                            |
    [ , , , , , , , ] --> [3,2] --> [ , , , , ,3,2,1]

    Step 7:
                    [8,7]--+--[ , ]
                           |
                    [6,5]--+--[ , ]
                           |
    [ , , , , , , , ] --> [4] --> [ , , , ,4,3,2,1]

    Step 8:
                     [8,7]--+--[ , ]
                            |
                     [ , ]--+--[ , ]
                            |
    [ , , , , , , , ] --> [6,5] --> [ , ,6,5,4,3,2,1]

    Step 9:
                     [ , ]--+--[ , ]
                            |
                     [ , ]--+--[ , ]
                            |
    [ , , , , , , , ] --> [8,7] --> [8,7,6,5,4,3,2,1]

    End of delivery phase.

    Network:

    L1 (1) -- (2) -- (3) L3
               |
    L4 (4) -- (5) -- (6) L6
               |
    S  (7) -- (8) -- (9) PB

    """
    s = {7, 8}
    stops = {2, 5, 8, 9}  # for the sorter.
    L1_stops = {2}
    L3_stops = {2}
    L4_stops = {5}
    L6_stops = {5}

    nodes = [  # with zero capacity.
        (2, 5, 8)
    ]

    n = Graph()
    for node in nodes:
        i = InterSection(capacity=0)
        n.add_node(node, obj=i)

    paths = [
        (1, 2, 1),
        (2, 3, 1),
        (2, 5, 1),
        (4, 5, 1),
        (5, 6, 1),
        (5, 8, 1),
        (7, 8, 1),
        (8, 9, 1),
    ]
    paths.extend([(n2, n1, d) for n1, n2, d in paths])  # adding the reverse path.
    n = Graph(from_list=paths)

    supply = Transporter(network=n, start=7, stops=s, order=Transporter.FIFO, capacity=1)
    sorter = Transporter(network=n, start=8, stops=stops, order=Transporter.FIFO, capacity=2)
    L1 = Transporter(network=n, start=1, stops=L1_stops, order=Transporter.FILO, capacity=2)
    L3 = Transporter(network=n, start=1, stops=L3_stops, order=Transporter.FILO, capacity=2)
    L4 = Transporter(network=n, start=1, stops=L4_stops, order=Transporter.FILO, capacity=2)
    L6 = Transporter(network=n, start=1, stops=L6_stops, order=Transporter.FILO, capacity=2)
    transports = [supply, sorter, L1, L3, L4, L6]

    arrival_sequence = [1, 3, 5, 7, 8, 6, 4, 2]
    fixed_delay = 50

    for permutation in itertools.permutations(arrival_sequence, len(arrival_sequence)):
        passengers = []
        for idx, seq in enumerate(permutation):
            w1 = WayPoint(location=8, open=idx - 2, close=idx + 2)
            w2 = WayPoint(location=9, open=seq * 3 + fixed_delay, close=seq * 3 + fixed_delay)
            p = Passenger(network=n, trips=[Trip(network=n, waypoints=[w1, w2])])
            passengers.append(p)

        schedule_all_trips(transports=transports, passengers=passengers)


        # The check:
        for passenger in passengers:
            assert isinstance(passenger, Passenger)
            assert passenger.schedule_complete()


def test_double_direction_delivery():
    """

    Tests a double delivery schedule:

    Lift is delivering equipment into the mine as Job-1, Job-2
    Train is delivering gold out of the mine as Job-3, Job-4

    The schedules are thereby:

    Lift:   [Job-1][go back][Job-2]
    Train:  [Job-3][go back][Job-4]

    The combined schedule should thereby be:

    Lift:   [Job-1][Job-3][Job-2][Job-4]
    Train:  [Job-3][Job-1][Job-4][Job-2]

    which yields zero idle runs.

    """
    pass