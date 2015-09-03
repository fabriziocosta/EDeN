
def add_stacking_base_pairs(graph_list=None):
    for g in graph_list:
        # iterate over nodes
        for n, d in g.nodes_iter(data=True):
            if d.get('position', False) == 0 or d.get('position', False) is not False:
                pos = d['position']
                # identify stacking neigbors
                # identify all neighbors
                neighbors = g.neighbors(n)
                if len(neighbors) >= 2:
                    # identify neighbors that have a greater 'position' attribute
                    greater_position_neighbors = [v for v in neighbors if g.node[v].get('position', False) and g.node[v]['position'] > pos]
                    if len(greater_position_neighbors) >= 2:  # there has to be at least a backbone vertex and a basepair vertex
                        # identify node that is connected by backbone edge
                        greater_position_neighbor_connected_by_backbone_list = [
                            v for v in greater_position_neighbors if g.edge[n][v]['type'] == 'backbone']
                        if len(greater_position_neighbor_connected_by_backbone_list) > 0:
                            greater_position_neighbor_connected_by_backbone = greater_position_neighbor_connected_by_backbone_list[0]  # take one
                            # identify node that is connected by basepair edge
                            greater_position_neighbor_connected_by_basepair_list = [
                                v for v in greater_position_neighbors if g.edge[n][v]['type'] == 'basepair']
                            if len(greater_position_neighbor_connected_by_basepair_list) > 0:
                                greater_position_neighbor_connected_by_basepair = greater_position_neighbor_connected_by_basepair_list[0]  # take one
                                # identify neighbor of greater_position_neighbor_connected_by_backbone
                                # that has greater position and is connected by basepair edge
                                greater_position_neighbor_connected_by_backbone_neighbors = g.neighbors(
                                    greater_position_neighbor_connected_by_backbone)
                                if len(greater_position_neighbor_connected_by_backbone_neighbors) > 0:
                                    greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair_list = [
                                        v for v in greater_position_neighbor_connected_by_backbone_neighbors if g.edge[greater_position_neighbor_connected_by_backbone][v]['type'] == 'basepair']
                                    if len(greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair_list) > 0:
                                        greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair = greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair_list[
                                            0]  # take one
                                        # check that
                                        # greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair
                                        # and greater_position_neighbor_connected_by_basepair are neighbors
                                        if greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair in g.neighbors(greater_position_neighbor_connected_by_basepair):
                                            # add vertex
                                            new_id = g.number_of_nodes()
                                            g.add_node(new_id)
                                            g.node[new_id]['label'] = 'o'
                                            g.node[new_id]['type'] = 'stack'
                                            # connect vertex
                                            g.add_edge(new_id, n)
                                            g.edge[new_id][n]['label'] = ':'
                                            g.edge[new_id][n]['type'] = 'stack'
                                            g.add_edge(new_id, greater_position_neighbor_connected_by_backbone)
                                            g.edge[new_id][greater_position_neighbor_connected_by_backbone]['label'] = ':'
                                            g.edge[new_id][greater_position_neighbor_connected_by_backbone]['type'] = 'stack'
                                            g.add_edge(new_id, greater_position_neighbor_connected_by_basepair)
                                            g.edge[new_id][greater_position_neighbor_connected_by_basepair]['label'] = ':'
                                            g.edge[new_id][greater_position_neighbor_connected_by_basepair]['type'] = 'stack'
                                            g.add_edge(
                                                new_id, greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair)
                                            g.edge[new_id][greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair][
                                                'label'] = ':'
                                            g.edge[new_id][greater_position_neighbor_connected_by_backbone_greater_position_neighbor_connected_by_basepair][
                                                'type'] = 'stack'
        yield g
