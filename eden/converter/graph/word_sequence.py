import networkx as nx
from eden import util


def word_sequence_to_eden(input=None, options=dict()):
    """
    Takes a list of strings, splits each string in words and yields networkx graphs.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    """

    for word_sequence in util.read(input):
        yield word_sequence_to_networkx(word_sequence)


def word_sequence_to_networkx(line):
    graph = nx.Graph()
    graph.graph['sequence'] = line
    for id, token in enumerate(unicode(line, errors='replace').split()):
        graph.add_node(id, label=token)
        if id > 0:
            graph.add_edge(id - 1, id, label='-')
    assert(len(graph) > 0), 'ERROR: generated empty graph. Perhaps wrong format?'
    return graph


def eden_to_word_sequence(input):
    for graph in input:
        seq = ""
        for n, d in graph.nodes_iter(data=True):
            seq += d['label'] + ' '
        yield seq
