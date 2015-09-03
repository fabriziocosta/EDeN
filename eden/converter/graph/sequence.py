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
    graph = nx.Graph()
    graph.graph['id'] = line
    for id, token in enumerate(line):
        graph.add_node(id, label=token)
        if id > 0:
            graph.add_edge(id - 1, id, label='-')
    assert(len(graph) > 0), 'ERROR: generated empty graph. Perhaps wrong format?'
    return graph
