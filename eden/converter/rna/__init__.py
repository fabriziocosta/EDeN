__author__ = "Fabrizio Costa, Bjoern Gruening"
__copyright__ = "Copyright 2014, Fabrizio Costa"
__credits__ = ["Fabrizio Costa", "Bjoern Gruening"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Fabrizio Costa"
__email__ = "costa@informatik.uni-freiburg.de"
__status__ = "Production"

import networkx as nx


def sequence_dotbracket_to_graph(seq_info=None, seq_struct=None):
	G = nx.Graph()
	lifo = list()
	for i,(c,b) in enumerate( zip(seq_info, seq_struct) ):
		G.add_node(i, label = c, position = i)
		if i > 0:
			G.add_edge(i,i-1, label='-', type='backbone')
		if b == '(':
			lifo += [i]
		if b == ')':
			j = lifo.pop()
			G.add_edge(i,j, label='=', type='basepair')
	return G