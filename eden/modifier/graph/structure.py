import networkx as nx
from eden.modifier.RNA import vertex_attributes
from collections import *


def edge_contraction(g, vertex_attribute = None):
	while True:
		change_has_occured = False
		for n, d in g.nodes_iter(data = True):
			if d.get(vertex_attribute,False) != False and (d.get('position',False) == 0 or d.get('position',False) != False):
				g.node[n]['label'] = g.node[n][vertex_attribute] 
				if d.get('contracted',False) == False:
					g.node[n]['contracted'] = set()
				g.node[n]['contracted'].add(n)
				neighbors = g.neighbors(n)
				if len(neighbors) > 0: 
					#identify neighbors that have a greater 'position' attribute and that have the same vertex_attribute
					greater_position_neighbors = [v for v in neighbors if g.node[v].get('position',False) and g.node[v].get(vertex_attribute,False) and g.node[v][vertex_attribute] == d[vertex_attribute] and g.node[v]['position'] > d['position'] ]
					if len(greater_position_neighbors) > 0 :
						#contract all neighbors
						#replicate all edges with n as endpoint instead of v
						cntr_edge_set = g.edges(greater_position_neighbors, data = True)
						new_edges = map(lambda x: (n,x[1],x[2]), cntr_edge_set)
						#remove nodes
						g.remove_nodes_from(greater_position_neighbors)
						#remode edges
						g.remove_edges_from(cntr_edge_set)
						#add edges if endpoint nodes still exist and they are not self loops
						new_valid_edges = [e for e in new_edges if e[1] in g.nodes() and e[1] != n ]
						g.add_edges_from(new_valid_edges) 
						#store neighbor ids in a list attribute	
						g.node[n]['contracted'].update(set(greater_position_neighbors))
						change_has_occured = True
						break
		if change_has_occured == False:
			break
	return g


def get_cumulative_weight(graph, vertex_list):
	weight = sum([graph.node[v].get('weight',1) for v in vertex_list])
	return weight


def get_dict_histogram_label(graph, vertex_list, bitmask):
	labels = [(abs(hash(graph.node[v].get('label','N/A'))) & bitmask) + 1 for v in vertex_list]
	dict_label = dict(Counter(labels).most_common())
	sparse_vec = {str(key):value for key,value in dict_label.iteritems()}
	return sparse_vec


def get_mode_label(graph, vertex_list):
	labels = [graph.node[v].get('label','N/A') for v in vertex_list]
	label = Counter(labels).most_common()[0][0]
	return label


def contraction(graph_list = None,  **options):
	level =  options.get('level',1)
	histogram_label =  options.get('histogram_label',False)
	mode_label =  options.get('mode_label',False)
	cumulative_weight =  options.get('cumulative_weight',True)
	nbits =  options.get('nbits',10)
	bitmask = pow(2, nbits) - 1

	#annotate with the adjacent edge labels the  
	for g in vertex_attributes.add_vertex_type(graph_list, level = level, output_attribute = 'type'):
		g_copy = g.copy()
		g_contracted = edge_contraction(g_copy, vertex_attribute = 'type')
		if mode_label or histogram_label or cumulative_weight:
			for n, d in g_contracted.nodes_iter(data = True):
				contracted = d.get('contracted',None)
				if contracted is None:
					raise Exception('Empty contraction list for: id %d data: %s' % (n,d))
				if mode_label :
					g_contracted.node[n]['label'] = get_mode_label(g,contracted)
				elif histogram_label :
					#update label with all contracted labels information using a histogram, i.e. a sparse vector 
					g_contracted.node[n]['label'] = get_dict_histogram_label(g,contracted, bitmask)
				if cumulative_weight :
					#update weight with the sum of all weights (or the count of vertices if the weight information is missing)
					g_contracted.node[n]['weight'] = get_cumulative_weight(g,contracted)
		yield g_contracted
