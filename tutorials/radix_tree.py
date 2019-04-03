from graph import Graph


__description__ = """
A radix tree (also radix trie or compact prefix tree) is a data structure that 
represents a space-optimized trie (prefix tree) in which each node that is the 
only child is merged with its parent. The result is that the number of children 
of every internal node is at most the radix r of the radix tree, where r is a 
positive integer and a power x of 2, having x ≥ 1. Unlike regular tries, edges 
can be labeled with sequences of elements as well as single elements. This makes 
radix trees much more efficient for small sets (especially if the strings are 
long) and for sets of strings that share long prefixes.

Unlike regular trees (where whole keys are compared en masse from their 
beginning up to the point of inequality), the key at each node is compared 
chunk-of-bits by chunk-of-bits, where the quantity of bits in that chunk at 
that node is the radix r of the radix trie. When the r is 2, the radix trie is 
binary (i.e., compare that node's 1-bit portion of the key), which minimizes 
sparseness at the expense of maximizing trie depth—i.e., maximizing up to 
conflation of nondiverging bit-strings in the key. When r is an integer power 
of 2 having r ≥ 4, then the radix trie is an r-ary trie, which lessens the 
depth of the radix trie at the expense of potential sparseness.

As an optimization, edge labels can be stored in constant size by using two 
pointers to a string (for the first and last elements).

[1] https://en.wikipedia.org/wiki/Radix_tree
"""


def radix_tree(datastructure):
    """
    Creates a radix tree from the data structure.
    :param datastructure: list of lists
    :return: graph.
    """
    assert isinstance(datastructure, list)
    assert all(isinstance(i, (tuple, list)) for i in datastructure)
    pass

