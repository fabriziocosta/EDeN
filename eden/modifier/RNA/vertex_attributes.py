import networkx as nx
import numpy as np 

def add_paired_unpaired_vertex_type(graph_list = None):
	for g in graph_list:
		#iterate over nodes
		for n, d in g.nodes_iter(data = True):
			g.node[n]['type'] = 'unpaired'
			for u,v,ed in g.edges_iter(n, data = True):
				if ed['type'] == 'basepair':
					g.node[n]['type'] = 'paired'
		yield g