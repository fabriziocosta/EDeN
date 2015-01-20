import networkx as nx
import numpy as np 

def add_paired_unpaired_vertex_type(graph_list = None, output_attribute = 'type'):
	for g in graph_list:
		#iterate over nodes
		for n, d in g.nodes_iter(data = True):
			g.node[n][output_attribute] = 'unpaired'
			for u,v,ed in g.edges_iter(n, data = True):
				if ed['type'] == 'basepair':
					g.node[n][output_attribute] = 'paired'
		yield g


def add_vertex_type(graph_list = None, output_attribute = 'type', separator = '', level = 1):
	"""
		level: int
			level = 1 considers all incident edges

			level = 2 considers all edges incident on the neighbors  
	"""
	for g in graph_list:
		#iterate over nodes
		for n, d in g.nodes_iter(data = True):
			#for all neighbors
			edge_labels = []
			if level == 1 :
				edge_labels += [ed.get('label','N/A') for u,v,ed in g.edges_iter(n, data = True)]
			elif level == 2 :
				neighbors = g.neighbors(n)
				for nn in neighbors:
					#extract list of edge labels
					edge_labels += [ed.get('label','N/A') for u,v,ed in g.edges_iter(nn, data = True)]
			else :
				raise Exception('Unknown level: %s' % level)
			#consider the sorted serialization of all labels as a type
			vertex_type = separator.join(sorted(edge_labels))
			g.node[n][output_attribute] = vertex_type
		yield g