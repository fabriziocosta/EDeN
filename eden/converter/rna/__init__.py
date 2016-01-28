import networkx as nx

__author__ = "Fabrizio Costa, Bjoern Gruening"
__copyright__ = "Copyright 2014, Fabrizio Costa"
__credits__ = ["Fabrizio Costa", "Bjoern Gruening"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Fabrizio Costa"
__email__ = "costa@informatik.uni-freiburgraph.de"
__status__ = "Production"


def sequence_dotbracket_to_graph(seq_info=None, seq_struct=None):
    """

    Parameters
    ----------
    seq_info string
        node labels eg a sequence string
    seq_struct  string
        dotbracket string

    Returns
    -------
        returns a nx.Graph
        secondary struct associated with seq_struct
    """
    graph = nx.Graph()
    lifo = list()
    for i, (c, b) in enumerate(zip(seq_info, seq_struct)):
        graph.add_node(i, label=c, position=i)
        if i > 0:
            graph.add_edge(i, i - 1, label='-', type='backbone', len=1)
        if b == '(':
            lifo.append(i)
        if b == ')':
            j = lifo.pop()
            graph.add_edge(i, j, label='=', type='basepair', len=1)
    return graph
