import networkx as nx
from eden.modifier.RNA import vertex_attributes



def edge_contraction(g, vertex_attribute = None):
	while True:
		change_has_occured = False
		for n, d in g.nodes_iter(data = True):
			if d.get(vertex_attribute,False) and d.get('position',False):
				neighbors = g.neighbors(n)
				if len(neighbors) > 0: 
					#identify neighbors that have a greater 'position' attribute and that have the same vertex_attribute
					greater_position_neighbors = [v for v in neighbors if g.node[v].get('position',False) and g.node[v].get(vertex_attribute,False) and g.node[v][vertex_attribute] == d[vertex_attribute] and g.node[v]['position'] > d['position'] ]
					if len(greater_position_neighbors) > 0 :
						#contract all neighbors
						#replicate all edges with n as endpoint instead of v
						cntr_edge_set = g.edges(greater_position_neighbors, data = True)
						g.add_edges_from(map(lambda x: (n,x[1],x[2]), cntr_edge_set)) 
						#remove nodes
						g.remove_nodes_from(greater_position_neighbors)
						#remode edges
						g.remove_edges_from(cntr_edge_set)
						#store neighbor ids in a list attribute	
						if d.get('contracted',False) == False:
							g.node[n]['contracted'] = greater_position_neighbors
						else:
							g.node[n]['contracted'] += greater_position_neighbors
						change_has_occured = True
						break
		if change_has_occured == False:
			break
	return g


def add_abstract_structure(graph_list = None):
	for g in vertex_attributes.add_paired_unpaired_vertex_type(graph_list):
		g_copy = g.copy()
		g_minor = edge_contraction(g_copy, vertex_attribute = 'type')
		#connect g_minor to original g 
		yield g_minor


def add_stacking_base_pairs(graph_list = None):
	for g in graph_list:
		#iterate over nodes
		for n, d in g.nodes_iter(data = True):
			if d.get('position',False):
				pos = d['position']
				#identify stacking neigbors
				#identify all neighbors
				neighbors = g.neighbors(n)
				if len(neighbors) >= 2: 
					#identify neighbors that have a greater 'position' attribute
					greater_position_neighbors = [v for v in neighbors if g.node[v].get('position',False) and g.node[v]['position'] > pos]
					if len(greater_position_neighbors) >= 2 : #there has to be at least a backbone vertex and a basepair vertex
						#identify node that is connected by backbone edge
						greater_position_neighbor_connected_by_backbone_list = [v for v in greater_position_neighbors if g.edge[n][v]['type'] == 'backbone']
						if len(greater_position_neighbor_connected_by_backbone_list) > 0 :
							greater_position_neighbor_connected_by_backbone = greater_position_neighbor_connected_by_backbone_list[0] #take one
							#identify node that is connected by basepair edge
							greater_position_neighbor_connected_by_basepair_list = [v for v in greater_position_neighbors if g.edge[n][v]['type'] == 'basepair']
							if len(greater_position_neighbor_connected_by_basepair_list) > 0 :
								greater_position_neighbor_connected_by_basepair = greater_position_neighbor_connected_by_basepair_list[0] #take one
								#identify neighbor of greater_position_neighbor_connected_by_backbone that has greater position and is connected by basepair edge
								greater_position_neighbor_connected_by_backbone_neighbors = g.neighbors(greater_position_neighbor_connected_by_backbone)
								if len(greater_position_neighbor_connected_by_backbone_neighbors) > 0 :
									greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair_list = [v for v in greater_position_neighbor_connected_by_backbone_neighbors if g.edge[greater_position_neighbor_connected_by_backbone][v]['type'] == 'basepair']
									if len(greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair_list) > 0 :
										greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair = greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair_list[0] #take one
										#check that greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair and greater_position_neighbor_connected_by_basepair are neighbors
										if greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair in g.neighbors(greater_position_neighbor_connected_by_basepair) :
											#add vertex
											new_id = g.number_of_nodes()
											g.add_node(new_id)
											g.node[new_id]['label'] = 'o'
											g.node[new_id]['type'] = 'stack'
											#connect vertex
											g.add_edge(new_id,n)
											g.edge[new_id][n]['label'] = ':'
											g.edge[new_id][n]['type'] = 'stack'
											g.add_edge(new_id,greater_position_neighbor_connected_by_backbone)
											g.edge[new_id][greater_position_neighbor_connected_by_backbone]['label'] = ':'
											g.edge[new_id][greater_position_neighbor_connected_by_backbone]['type'] = 'stack'
											g.add_edge(new_id,greater_position_neighbor_connected_by_basepair)
											g.edge[new_id][greater_position_neighbor_connected_by_basepair]['label'] = ':'
											g.edge[new_id][greater_position_neighbor_connected_by_basepair]['type'] = 'stack'
											g.add_edge(new_id,greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair)
											g.edge[new_id][greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair]['label'] = ':'
											g.edge[new_id][greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair]['type'] = 'stack'
		yield g