import networkx as nx
from eden import util

def word_sequence_to_eden(input = None, options = dict()):
    """
    Takes a list of strings, splits each string in words and yields networkx graphs.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    """

    for word_sequence in util.read( input ):
        yield word_sequence_to_networkx(word_sequence)

def word_sequence_to_networkx(line):
    G = nx.Graph()
    for id,token in enumerate(unicode(line, errors = 'replace').split()):
        G.add_node(id, label = token)
        if id > 0:
            G.add_edge(id-1, id, label = '-')
    assert(len(G) > 0),'ERROR: generated empty graph. Perhaps wrong format?'
    return G
