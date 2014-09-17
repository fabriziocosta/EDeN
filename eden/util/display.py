import networkx as nx
import pylab as plt

def draw_graph(graph, vertex_label='label', edge_label='label', vertex_color='', size=10, layout='graphviz', node_size=600):
    
    plt.figure(figsize=(size,size))
    plt.grid(False)
    plt.axis('off')
    
    vertex_labels=dict([(u,d[vertex_label]) for u,d in graph.nodes(data=True)])
    edge_labels=dict([((u,v,),d[edge_label]) for u,v,d in graph.edges(data=True)])
    
    if vertex_color == '':
        node_color = 'white'
    else:
        node_color=[graph.node[u][vertex_color] for u in graph.nodes()]
    if layout == 'graphviz':
        pos = nx.graphviz_layout(graph)
    else:
        pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos,node_color=node_color,
        alpha=0.6,node_size=node_size, cmap = plt.get_cmap('YlOrRd'))
    nx.draw_networkx_labels(graph,pos, vertex_labels, font_size=9,font_color='black')
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()