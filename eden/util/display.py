import networkx as nx
import pylab as plt
import json
from networkx.readwrite import json_graph

def draw_graph( graph, 
    vertex_label = 'label', 
    secondary_vertex_label = None, 
    edge_label = 'label', 
    secondary_edge_label = None, 
    vertex_color = '', 
    vertex_alpha = 0.6,
    size = 10,
    size_x_to_y_ratio = 1, 
    node_size = 600,
    font_size = 9,
    layout = 'graphviz', 
    prog  =  'neato',
    node_border = False,
    colormap = 'YlOrRd',
    invert_colormap = False,
    verbose = True ):
    

    size_x = size
    size_y = int(float(size) / size_x_to_y_ratio)

    plt.figure( figsize = ( size_x,size_y ) )
    plt.grid( False )
    plt.axis( 'off' )
    
    if secondary_vertex_label:
        vertex_labels = dict( [( u,'%s\n%s'%( d.get( vertex_label,'N/A' ),d.get( secondary_vertex_label,'N/A' )  )  ) for u,d in graph.nodes( data = True )] )
    else:
        vertex_labels = dict( [( u,d.get( vertex_label,'N/A' ) ) for u,d in graph.nodes( data = True ) ] )
    
    edges_normal = [( u,v ) for ( u,v,d ) in graph.edges( data = True ) if d.get( 'nesting', False ) == False]
    edges_nesting = [( u,v ) for ( u,v,d ) in graph.edges( data = True ) if d.get( 'nesting', False ) == True]

    if secondary_edge_label:
        edge_labels = dict( [( ( u,v, ),'%s\n%s'%( d.get( edge_label,'N/A' ),d.get( secondary_edge_label,'N/A' ) )  ) for u,v,d in graph.edges( data = True )] )
    else:
        edge_labels = dict( [( ( u,v, ),d.get( edge_label,'N/A' )  ) for u,v,d in graph.edges( data = True )] )

    if vertex_color == '':
        node_color  =  'white'
    else:
        if invert_colormap:
            node_color = [ - d.get( vertex_color,0 ) for u,d in graph.nodes( data = True ) ]
        else:
            node_color = [ d.get( vertex_color,0 ) for u,d in graph.nodes( data = True ) ]

    if layout == 'graphviz':
        pos  =  nx.graphviz_layout( graph, prog  =  prog )
    elif layout == 'circular':
        pos  =  nx.circular_layout( graph )
    elif layout == 'random':
        pos  =  nx.random_layout( graph )
    elif layout == 'spring':
        pos  =  nx.spring_layout( graph )
    elif layout == 'shell':
        pos  =  nx.shell_layout( graph )
    elif layout == 'spectral':
        pos  =  nx.spectral_layout( graph )
    else:
        raise Exception( 'Unknown layout format: %s' % layout )

    if node_border == False :
        linewidths  =  0.001
    else:
        linewidths  =  1

    nx.draw_networkx_nodes( graph,pos,
        node_color  =  node_color,
        alpha  =  vertex_alpha,
        node_size  =  node_size, 
        linewidths  =  linewidths,
        cmap  =  plt.get_cmap( colormap ) )
    nx.draw_networkx_labels( graph,pos, vertex_labels, font_size  =  font_size,font_color  =  'black' )
    nx.draw_networkx_edges( graph, pos, 
        edgelist  =  edges_normal, 
        width  =  2, 
        edge_color  =  'k', 
        alpha  =  0.5 )
    nx.draw_networkx_edges( graph, pos, 
        edgelist  =  edges_nesting, 
        width  =  1, 
        edge_color  =  'k', 
        style  =  'dashed', 
        alpha  =  0.5 )
    nx.draw_networkx_edge_labels( graph, pos, edge_labels  =  edge_labels, font_size  =  font_size )
    if verbose:
        title = str(graph.graph.get('id','')) + "\n" + str(graph.graph.get('info',''))
        plt.title(title)
    plt.show()


def draw_adjacency_graph ( A,
    node_color  =  None, 
    size  =  10,
    layout  =  'graphviz', 
    prog  =  'neato',
    node_size  =  80,
    colormap  =  'autumn' ):

    graph  =  nx.from_scipy_sparse_matrix( A )

    plt.figure( figsize  =  ( size,size ) )
    plt.grid( False )
    plt.axis( 'off' )

    if layout == 'graphviz':
        pos  =  nx.graphviz_layout( graph, prog  =  prog )
    else:
        pos  =  nx.spring_layout( graph )

    if  len( node_color ) == 0:
        node_color  =  'gray'
    nx.draw_networkx_nodes( graph, pos,
                           node_color  =  node_color, 
                           alpha  =  0.6, 
                           node_size  =  node_size, 
                           cmap  =  plt.get_cmap( colormap ) )
    nx.draw_networkx_edges( graph, pos, alpha  =  0.5 )
    plt.show(  )


class SetEncoder( json.JSONEncoder ):
    def default( self, obj ):
        if isinstance( obj, set ):
            return list( obj )
        return json.JSONEncoder.default( self, obj )


def serialize_graph( graph ):
    json_data  =  json_graph.node_link_data( graph )
    serial_data  =  json.dumps( json_data, separators  =  ( ',',':' ), indent  =  4, cls  =  SetEncoder )
    return serial_data


def embed2D(data, vectorizer, size = 10, n_components = 5, gamma = 20, nu = 0.01, n_jobs = 1, colormap = 'YlOrRd'):
    import numpy as np
    if hasattr( data, '__iter__' ):
        iterable = data
    else: 
        raise Exception( 'ERROR: Input must be iterable' )
    import itertools
    iterable_1,iterable_2 = itertools.tee(iterable)
    #get labels
    labels=[]
    for graph in iterable_2:
        label = graph.graph.get('id',None)
        if label:
            labels.append( label )

    #transform iterable into sparse vectors
    X = vectorizer.transform( iterable_1 , n_jobs = n_jobs )

    #embed high dimensional sparse vectors in 2D
    from sklearn import metrics
    D = metrics.pairwise.pairwise_distances( X )

    from sklearn.manifold import MDS
    feature_map = MDS( n_components=n_components, dissimilarity='precomputed')
    X_explicit=feature_map.fit_transform( D )

    from sklearn.decomposition import TruncatedSVD
    pca = TruncatedSVD( n_components = 2 )
    X_reduced = pca.fit_transform( X_explicit )

    plt.figure(figsize=(size,size))

    #make mesh
    x_min, x_max = X_reduced[:, 0].min(), X_reduced[:, 0].max()
    y_min, y_max = X_reduced[:, 1].min(), X_reduced[:, 1].max()
    step_num = 50
    h = min( ( x_max - x_min ) / step_num  , ( y_max - y_min ) / step_num )# step size in the mesh
    b = h * 10 # border size
    x_min, x_max = X_reduced[:, 0].min() - b, X_reduced[:, 0].max() + b
    y_min, y_max = X_reduced[:, 1].min() - b, X_reduced[:, 1].max() + b
    xx, yy = np.meshgrid( np.arange( x_min, x_max, h ), np.arange( y_min, y_max, h ) )

    #induce a one class model to estimate densities
    from sklearn.svm import OneClassSVM
    clf = OneClassSVM( gamma = gamma, nu = nu )
    clf.fit( X_reduced )

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max] . [y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    # Put the result into a color plot
    levels = np.linspace(min(Z), max(Z), 40)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap = plt.get_cmap( colormap ), alpha = 0.9, levels = levels)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                alpha=.5, 
                s=70, 
                edgecolors='none', 
                c = 'white',
                cmap = plt.get_cmap('YlOrRd'))
    #labels
    for id in range( X_reduced.shape[0] ):
        if labels is None:
            label = str(id) 
        else:
            label = labels[id] 
        x = X_reduced[id, 0]
        y = X_reduced[id, 1]
        plt.annotate(label,xy = (x,y), xytext = (0, 0), textcoords = 'offset points')
    plt.show()

def dendrogram(data, vectorizer, color_threshold=1, size = 10, n_jobs = 1):
    import numpy as np
    if hasattr( data, '__iter__' ):
        iterable = data
    else: 
        raise Exception( 'ERROR: Input must be iterable' )
    import itertools
    iterable_1,iterable_2 = itertools.tee(iterable)
    #get labels
    labels=[]
    for graph in iterable_2:
        label = graph.graph.get('id',None)
        if label:
            labels.append( label )
    #transform input into sparse vectors
    X = vectorizer.transform( iterable_1 , n_jobs = n_jobs )
    
    #labels
    if not labels:
        labels = [str(i) for i in range(X.shape[0] )]

    #embed high dimensional sparse vectors in 2D
    from sklearn import metrics
    from scipy.cluster.hierarchy import linkage, dendrogram
    D = metrics.pairwise.pairwise_distances(X)
    Z = linkage(D)
    plt.figure(figsize=(size, size))
    dendrogram(Z, color_threshold=color_threshold, labels=labels, orientation='right')
    plt.show()


def KernelQuickShiftTreeEmbedding(X, order=1, metric='linear', **args):
    #extract pairwise similarity matrix with desired kernel
    from sklearn import metrics
    K=metrics.pairwise.pairwise_kernels(X, metric=metric,**args)
    
    #compute instance density as average pairwise similarity
    import numpy as np
    density=np.sum(K,0)/K.shape[0]
    
    #compute list of nearest neighbors
    Ka=np.argsort(-K)
    
    #compute density for each nearest neighbor
    Kad=density[Ka]

    import networkx as nx
    G=nx.Graph()
    G.add_nodes_from(range(K.shape[0]))
    #for all instances
    for i,row in enumerate(Kad): 
        i_density=row[0]
        k_nearest_denser_neighbors=[]
        #for all neighbors from the closest to the furthest
        for jj,d in enumerate(row):
            j=Ka[i,jj]
            if jj>0:
                j_density=d
                #if the density of the neighbor is higher than the density of the instance
                if j_density > i_density:
                    #store density, similarity, identity of neighbor
                    k_nearest_denser_neighbors.append((j_density,K[i,j],j))
                    #when we have identified n=order neighbors that have a greater density then the instance 
                    if len(k_nearest_denser_neighbors) >= order:
                        #find the one with the highest density
                        j_density_max,k_max, j_max = max(k_nearest_denser_neighbors)
                        #add a link between the instance and the higher density neighbor
                        G.add_edge(i,j_max,weight=K[i,j])
                        k_nearest_denser_neighbors=[]
                        break
        #in case we did not manage to find n=order neighbors with higher density 
        if k_nearest_denser_neighbors:
            j_density_max,k_max, j_max = max(k_nearest_denser_neighbors)
            G.add_edge(i,j_max,weight=K[i,j])
    #make a fast spring layout of the resulting tree
    X_ = nx.graphviz_layout(G,prog='sfdp')
    #extract the coordinates of the embedding
    X_2D=[]
    for i in range(K.shape[0]):
        X_2D.append(list(X_[i]))
    X_emb = np.array(X_2D)
    return X_emb