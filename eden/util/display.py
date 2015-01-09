import networkx as nx
import pylab as plt

def draw_graph(graph, 
    vertex_label='label', 
    secondary_vertex_label='', 
    edge_label='label', 
    secondary_edge_label='', 
    vertex_color='', 
    size=10, 
    layout='graphviz', 
    prog = 'neato',
    node_size=600,
    node_border=True):
    
    plt.figure(figsize=(size,size))
    plt.grid(False)
    plt.axis('off')
    
    if secondary_vertex_label:
        vertex_labels=dict([(u,'%s\n%s'%(d.get(vertex_label,'N/A'),d.get(secondary_vertex_label,'N/A') ) ) for u,d in graph.nodes(data=True)])
    else:
        vertex_labels=dict([(u,d.get(vertex_label,'N/A')) for u,d in graph.nodes(data=True) ])
    
    edges_normal=[(u,v) for (u,v,d) in graph.edges(data=True) if d.get('nesting', False) == False]
    edges_nesting=[(u,v) for (u,v,d) in graph.edges(data=True) if d.get('nesting', False) == True]

    if secondary_edge_label:
        edge_labels=dict([((u,v,),'%s\n%s'%(d.get(edge_label,'N/A'),d.get(secondary_edge_label,'N/A')) ) for u,v,d in graph.edges(data=True)])
    else:
        edge_labels=dict([((u,v,),d.get(edge_label,'N/A') ) for u,v,d in graph.edges(data=True)])

    if vertex_color == '':
        node_color = 'white'
    else:
        node_color=[graph.node[u][vertex_color] for u in graph.nodes()]

    if layout == 'graphviz':
        pos = nx.graphviz_layout(graph, prog = prog)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    elif layout == 'random':
        pos = nx.random_layout(graph)
    elif layout == 'spring':
        pos = nx.spring_layout(graph)
    elif layout == 'shell':
        pos = nx.shell_layout(graph)
    elif layout == 'spectral':
        pos = nx.spectral_layout(graph)
    else:
        raise Exception('Unknown layout format: %s' % layout)

    if node_border == False :
        linewidths = 0.001
    else:
        linewidths = 1

    nx.draw_networkx_nodes(graph,pos,
        node_color=node_color,
        alpha=0.6,
        node_size=node_size, 
        linewidths=linewidths,
        cmap = plt.get_cmap('YlOrRd'))
    nx.draw_networkx_labels(graph,pos, vertex_labels, font_size = 9,font_color = 'black')
    nx.draw_networkx_edges(graph, pos, 
        edgelist = edges_normal, 
        width = 2, 
        edge_color = 'b', 
        alpha=0.5)
    nx.draw_networkx_edges(graph, pos, 
        edgelist = edges_nesting, 
        width = 1, 
        edge_color = 'b', 
        style = 'dashed', 
        alpha=0.5)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()


def draw_adjacency_graph (A,
    node_color = None, 
    size = 10,
    layout = 'graphviz', 
    prog = 'neato',
    node_size = 80):

    graph = nx.from_scipy_sparse_matrix(A)

    plt.figure(figsize=(size,size))
    plt.grid(False)
    plt.axis('off')

    if layout == 'graphviz':
        pos = nx.graphviz_layout(graph, prog = prog)
    else:
        pos = nx.spring_layout(graph)

    if  len(node_color) == 0:
        node_color='gray'
    nx.draw_networkx_nodes(graph, pos,
                           node_color = node_color, 
                           alpha = 0.6, 
                           node_size = node_size, 
                           cmap = plt.get_cmap('autumn'))
    nx.draw_networkx_edges(graph, pos, alpha = 0.5)
    plt.show()

def serialize_graph(graph):
    import json
    from networkx.readwrite import json_graph
    json_data = json_graph.node_link_data(graph)
    serial_data = json.dumps(json_data, separators=(',',':'), indent=4)
    return serial_data