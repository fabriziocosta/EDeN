import networkx as nx
from collections import Counter, namedtuple

def edge_contraction(graph = None, node_attribute = None):
	g = graph.copy()
	for n, d in g.nodes_iter(data = True):
		g.node[n]['contracted'] = set()
	while True:
		change_has_occured = False
		for n, d in g.nodes_iter(data = True):
			g.node[n]['label'] = g.node[n][node_attribute] 
			if d.get(node_attribute,False) != False and (d.get('position',False) == 0 or d.get('position',False) != False):	
				if d.get('contracted',False) == False:
					g.node[n]['contracted'] = set()
				g.node[n]['contracted'].add(n)
				neighbors = g.neighbors(n)
				if len(neighbors) > 0: 
					#identify neighbors that have a greater 'position' attribute and that have the same node_attribute
					greater_position_neighbors = [v for v in neighbors if g.node[v].get('position',False) and g.node[v].get(node_attribute,False) and g.node[v][node_attribute] == d[node_attribute] and g.node[v]['position'] > d['position'] ]
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

def contraction_histogram(input_attribute = None, graph = None, id_nodes = None):
	labels = [graph.node[v].get(input_attribute,'N/A') for v in id_nodes]
	dict_label = dict(Counter(labels).most_common())
	sparse_vec = {str(key):value for key,value in dict_label.iteritems()}
	return sparse_vec

def contraction_sum(input_attribute = None, graph = None, id_nodes = None):
	vals = [float(graph.node[v].get(input_attribute,1)) for v in id_nodes]
	return sum(vals)

def contraction_average(input_attribute = None, graph = None, id_nodes = None):
	vals = [float(graph.node[v].get(input_attribute,0)) for v in id_nodes]
	return sum(vals)/float(len(vals))

def contraction_categorical(input_attribute = None, graph = None, id_nodes = None, separator = '.'):
	vals = sorted([str(graph.node[v].get(input_attribute,'N/A')) for v in id_nodes])
	return separator.join(vals)

def contraction_set_categorical(input_attribute = None, graph = None, id_nodes = None, separator = '.'):
	vals = sorted(set([str(graph.node[v].get(input_attribute,'N/A')) for v in id_nodes]))
	return separator.join(vals)


contraction_modifer_map = {'histogram':contraction_histogram, 
	'sum':contraction_sum, 
	'average':contraction_average, 
	'categorical':contraction_categorical,
	'set_categorical': contraction_set_categorical}
contraction_modifier = namedtuple('modifier', 'input output action')
label_modifier = contraction_modifier(input='type', output='label', action='set_categorical')
weight_modifier = contraction_modifier(input='weight', output='weight', action='sum')
modifiers = [label_modifier, weight_modifier]


def contraction(graphs = None,  contraction_attribute = 'label', modifiers = modifiers, **options):
	''' 
		modifiers: list of dictionaries, each containing the keys: input, output and action.
		"input" identifies the node attribute that is extracted from all contracted nodes.
		"output" identifies the node attribute that is written in the resulting graph.
		"action" is one of the following reduction operations: 
		1. histogram, 
		2. sum, 
		3. average, 
		4. categorical, 
		5. set_categorical.
		"histogram" returns a sparse vector with numerical hased keys, "sum" and "average" cast the values into floats before
		computing the sum and average respectively, "categorical" returns the concatenation string of the
		lexicographically sorted list of input attributes, "set_categorical" returns the concatenation string of the
		lexicographically sorted set of input attributes.  
	'''
	nbits =  options.get('nbits',10)
	bitmask = pow(2, nbits) - 1

	for g in graphs:
		g_contracted = edge_contraction(graph = g, node_attribute = contraction_attribute)
		for n, d in g_contracted.nodes_iter(data = True):
			contracted = d.get('contracted',None)
			if contracted is None:
				raise Exception('Empty contraction list for: id %d data: %s' % (n,d))
			#store the dictionary of all contracted nodes dictionaries 
			#g_contracted.node[n]['contracted_dicts'] = {v:g.node[v] for v in contracted}
			#process the action 
			for modifier in modifiers:
				modifier_func = contraction_modifer_map[modifier.action]
				g_contracted.node[n][modifier.output] = modifier_func(input_attribute = modifier.input, graph = g, id_nodes = contracted)
		yield g_contracted