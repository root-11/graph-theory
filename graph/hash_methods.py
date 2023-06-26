import hashlib

from graph import Graph


def graph_hash(graph):
    """ Generates the top hash of the graph using sha3_256.
    :param graph: instance of class Graph.
    :return: graph hash (int) and graph (Graph) with hash values
    """
    assert isinstance(graph, Graph)
    hash_func = hashlib.sha3_256()
    nodes = bytes("|".join(str(n) for n in sorted(graph.nodes())), 'utf-8')
    edges = bytes("|".join(str(e) for e in sorted(graph.edges())), 'utf-8')
    hash_func.update(nodes + edges)
    return int(hash_func.hexdigest(), 16)


def flow_graph_hash(graph):
    """
    Calculates the hash of a flow graph, where the properties of the nodes
    and the supplying nodes are included in the hash.

    Any upstream change in hash will thereby propagate downstream.
    """
    assert isinstance(graph, Graph)
    sources = graph.nodes(in_degree=0)

    original_hash = 'original hash'
    new_hash = 'new_hash'
    hash_graph = Graph()  # new graph with hashes.
    visited = set()

    while sources:
        source = sources[0]
        sources = sources[1:]

        suppliers = graph.nodes(to_node=source)

        hash_func = hashlib.sha3_256()
        hash_func.update(bytes(str(source), 'utf-8'))
        for supplier in suppliers:
            if graph.depth_first_search(start=source, end=supplier):
                continue  # it's a cycle.
            d = hash_graph.node(supplier)
            hash_func.update(bytes(d[new_hash], 'utf-8'))
        source_hash = hash_func.hexdigest()

        if source not in hash_graph:
            obj = {original_hash: source, new_hash: source_hash}
            hash_graph.add_node(source, obj=obj)
        else:
            n = hash_graph.node(source)
            n[new_hash] = source_hash

        receivers = graph.nodes(from_node=source)
        for receiver in receivers:
            if receiver in visited:
                continue
            visited.add(receiver)

            if receiver not in hash_graph:
                obj = {original_hash: receiver, new_hash: None}
                hash_graph.add_node(node_id=receiver, obj=obj)
            hash_graph.add_edge(source, receiver)
            if receiver not in sources:
                sources.append(receiver)

    for sink in graph.nodes(out_degree=0):
        n = hash_graph.node(sink)
        assert n[new_hash] is not None, n

    return hash_graph


def merkle_tree(data_blocks):
    """
    A hash tree or Merkle tree is a tree in which every leaf node is labelled with
    the hash of a data block, and every non-leaf node is labelled with the
    cryptographic hash of the labels of its child nodes. Hash trees allow efficient
    and secure verification of the contents of large data structures. Hash trees
    are a generalization of hash lists and hash chains.

                        Top Hash
                     hash ( 0 + 1 )
                       ^      ^
                       |      |
               +------->      +---------+
               ^                        ^
               |                        |
               +                        +
            Hash 0                    Hash 1
      hash ( 0-0 + 0-1 )        hash ( 1-0 + 1-1 )
         ^           ^           ^           ^
         |           |           |           |
         +           +           +           +
      Hash 0-0    Hash 0-1    Hash 1-0     Hash 1-1
      hash(L1)    hash(L2)    hash(L3)     hash(L4)
         ^           ^           ^           ^
         |           |           |           |
    +----------------------------------------------+
    |    L1          L2          L3          L4    | Data blocks
    +----------------------------------------------+

    """
    g = Graph()

    # initial hash:
    leaves = []
    for block in data_blocks:
        assert isinstance(block, bytes)
        hash_func = hashlib.sha3_256()
        hash_func.update(block)
        uuid = hash_func.hexdigest()
        leaves.append(uuid)
        g.add_node(node_id=uuid)

    # populate graph
    while leaves:
        if len(leaves) == 1:
            return g  # <--- point of return.

        c1, c2 = leaves[:2]
        leaves = leaves[2:]

        hash_func = hashlib.sha3_256()
        hash_func.update(bytes(c1, 'utf-8') + bytes(c2, 'utf-8'))
        uuid = hash_func.hexdigest()
        leaves.append(uuid)
        g.add_node(node_id=uuid)
        g.add_edge(c1, uuid)
        g.add_edge(c2, uuid)