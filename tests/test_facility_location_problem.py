from tests.test_graph import london_underground
"""
The problem of deciding the exact place in a community where a school or a fire station should be located,
is classified as the facility location problem.

If the facility is a school, it is desirable to locate it so that the sum 
of distances travelled by all members of the communty is as short as possible. 
This is the minimum of sum - or in short `minsum` of the graph.

If the facility is a firestation, it is desirable to locate it so that the distance from the firestation
to the farthest point in the community is minimized. 
This is the minimum of max distances - or in short `minmax` of the graph.
"""


def test_minsum():
    g = london_underground()
    stations = g.minsum()
    assert len(stations) == 1
    station_list = [g.node(s) for s in stations]
    assert station_list == [(51.515, -0.1415, 'Oxford Circus')]


def test_minmax():
    g = london_underground()
    stations = g.minmax()
    assert len(stations) == 3
    station_list = [g.node(s) for s in stations]
    assert station_list == [(51.5226, -0.1571, 'Baker Street'),
                            (51.5142, -0.1494, 'Bond Street'),
                            (51.5234, -0.1466, "Regent's Park")]
