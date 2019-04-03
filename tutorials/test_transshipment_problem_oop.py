from graph import Graph
from tutorials.transshipment_problem_oop import Traveller, Transporter, schedule_all_trips


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
    pass


def test_transports_filo():
    """
    passengers must move to the back of the bus to let other passengers on
    board. If a passenger at the rear of the bus wants to get out, all
    passengers in front of him/her must get off the bus first.
    """
    pass


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
        Traveller(network=network, trips=[(1, 2, 0, float('inf'))]),
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
        Traveller(network=network, trips=[(1, 2, 0, float('inf'))]),
        Traveller(network=network, trips=[(2, 3, 0, float('inf'))]),
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
        Traveller(network=network, trips=[(1, 2, 0, float('inf'))]),
        Traveller(network=network, trips=[(2, 3, 0, float('inf'))]),
        Traveller(network=network, trips=[(3, 4, 0, float('inf'))]),
        Traveller(network=network, trips=[(4, 1, 0, float('inf'))])
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
        Traveller(network=network, trips=[('airport', 1, 0, float('inf'))]),
        Traveller(network=network, trips=[('airport', 2, 0, float('inf'))]),
        Traveller(network=network, trips=[('airport', 3, 0, float('inf'))]),
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
        Traveller(network=network, trips=[(1, 'airport', 0, float('inf'))]),
        Traveller(network=network, trips=[(2, 'airport', 0, float('inf'))]),
        Traveller(network=network, trips=[(3, 'airport', 0, float('inf'))]),
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
        Traveller(network=network, trips=[(1, 'airport', 0, float('inf'))]),
        Traveller(network=network, trips=[('airport', 1, 0, float('inf'))]),
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
        Traveller(network=network, trips=[('airport', 1, 0, float('inf'))]),
        Traveller(network=network, trips=[('airport', 2, 0, float('inf'))]),
        Traveller(network=network, trips=[('airport', 3, 0, float('inf'))]),
        Traveller(network=network, trips=[(1, 'airport', 0, float('inf'))]),
        Traveller(network=network, trips=[(2, 'airport', 0, float('inf'))]),
        Traveller(network=network, trips=[(3, 'airport', 0, float('inf'))]),
    ]

    transports = [
        Transporter(network=network, start='airport', stops=set(network.nodes())),
    ]

    schedule_all_trips(transports=transports, passengers=passengers)


def test_gold_delivery_to_surface():
    """
    Assures that gold dug out of the mine can be returned to the surface.
    """
    pass


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