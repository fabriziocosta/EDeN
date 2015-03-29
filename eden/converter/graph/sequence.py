import networkx as nx
from eden.util import read


def sequence_to_eden(input=None, options=dict()):
    """
    Takes a list of strings and yields networkx graphs.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    """

    for sequence in read(input):
        yield sequence_to_networkx(sequence)


def sequence_to_networkx(line):
    G = nx.Graph()
    G.graph['id'] = line
    for id, token in enumerate(line):
        G.add_node(id, label=token)
        if id > 0:
            G.add_edge(id - 1, id, label='-')
    assert(len(G) > 0), 'ERROR: generated empty graph. Perhaps wrong format?'
    return G
