from graph import Graph
from tutorials.hashgraph import merkle_tree, flow_graph_hash, graph_hash


def test_merkle_tree_1_block():
    data_blocks = [b"this"]
    g = merkle_tree(data_blocks)
    assert len(g.nodes()) == 1


def test_merkle_tree_2_blocks():
    data_blocks = [b"this",
                   b"that"]
    g = merkle_tree(data_blocks)
    assert len(g.nodes()) == 3


def test_merkle_tree_3_blocks():
    data_blocks = [b"this",
                   b"that",
                   b"them"]
    g = merkle_tree(data_blocks)
    assert len(g.nodes()) == 5


def test_merkle_tree_4_blocks():
    data_blocks = [b"this",
                   b"that",
                   b"them",
                   b"they"]
    g = merkle_tree(data_blocks)
    assert len(g.nodes()) == 7


def test_flow_graph_hash_01():
    """
    This example includes a loop to distinguish it from the common merkle tree.

    S-1         S-2             S-3                 S-4
 (hash S1)   (hash S2)       (hash S3)           (hash S4)
     +          +   +            +
     |          |   +----------->+
     |          |                +<-------------+
     v          v                v              |
          I-1                    I-2            | (loop)
     (hash S1+S2+I1)         (hash S3 + I2)     |
     +          +                +              |
     |          |                +------------->+
     v          |                |
     E-1        +---> E-2 <------+
 (hash I1+E1)    (hash I1+I2+E2)

    """
    L = [
        ('s-1', 'i-1', 1),
        ('s-2', 'i-1', 1),
        ('i-1', 'e-1', 1),
        ('i-1', 'e-2', 1),
        ('s-3', 'i-2', 1),
        ('i-2', 'i-2', 1),
        ('i-2', 'e-2', 1),
    ]
    g = Graph(from_list=L)
    g.add_node('s-4')
    g2 = flow_graph_hash(g)
    assert len(g2) == len(g)


def test_flow_graph_loop_01():
    L = [
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (3, 2, 1)
    ]
    g = Graph(from_list=L)
    g2 = flow_graph_hash(g)
    assert len(g2) == len(g)


def test_flow_graph_async_01():
    """

    (s1) --> (i2) --> (e4)
                      /
             (s3) -->/
    """
    L = [
        (1, 2, 1),
        (2, 4, 1),
        (3, 4, 1)
    ]
    g = Graph(from_list=L)
    g2 = flow_graph_hash(g)
    assert len(g2) == len(g)


def test_graph_hash():
    """
    Simple test of the graph hash function.
    """
    L = [
        (1, 2, 1),
        (2, 3, 1),
        (3, 4, 1),
        (3, 2, 1)
    ]
    g = Graph(from_list=L)
    h = graph_hash(g)
    assert isinstance(h, int)
    assert sum((int(d) for d in str(h))) == 312

