def add_paired_unpaired_vertex_type(graph_list=None, output_attribute='type'):
    for g in graph_list:
        # iterate over nodes
        for n, d in g.nodes_iter(data=True):
            g.node[n][output_attribute] = 'unpaired'
            for u, v, ed in g.edges_iter(n, data=True):
                if ed['type'] == 'basepair':
                    g.node[n][output_attribute] = 'paired'
        yield g
