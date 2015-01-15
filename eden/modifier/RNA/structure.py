import networkx as nx

def add_stacking_base_pairs(graph_list = None):
	for g in graph_list:
		#iterate over nodes
		for n, d in g.nodes_iter(data = True):
			#identify neighbors
			#identify stacking neigbors
			#add vertex
			#connect vertex
		yield g