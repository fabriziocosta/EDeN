import networkx as nx
import subprocess as sp


#each node when it is created should have a comment that describes it
#each node must have the 'label' attribute
#the presence of attributes should be tested with if dict.get('attr',False)
#the procedure to color nodes should be a separate function, i.e. colors should not be assigned during graph construction

def _add_stacks(G,start,optlist): 
    '''
    Create supporting 'p' nodes in stacking base-pairs
    '''
    #interesting edges:
    ie = [ a for a in G.edges() if a[0]-a[1] != -1 and a[0] > start ]
    nextnode = len(G)
    csum = 0
    for (a,b) in ie:
        if a+b == csum:
            #found a stacking oO
            # actually add stacks.. 
            if optlist['stack']:
                G.add_node(nextnode)
                G.node[nextnode]['label'] = 'P'
                for i in [a-1,a,b,b+1]:
                    G.add_edge(i,nextnode)
                nextnode += 1
        else:
            csum = a+b


def _add_abstr(G,start,end,optlist):
    '''
    we add some nodes that indicate the structure element a base belongs to.

    1. most often start and end will mark the beginning of a stacking.. so we walk along the stacking first. 
         
    2. then we walk along the unpaired bases until we find an other stacking in which case: recurse. 

    .. in the end we find out what this structure actually is by seing how many times we found other stackings.

    we also connect this element with the children found in the recursive call.. '''
    
    # interesting edges
    ie = [ e for e in G.edges() if e[0]-e[1] != -1 and e[0]>start and e[1] <= end] 
    # we know that start binds to end so we do ths:
    i = start+1
    j = end-1
    k = 0 # k is the index in ie
    for (b,e) in ie:
        if (b,e) != (i,j):
            break
        else:
            i += 1
            j -= 1
        k += 1
    if k == len(ie): #overflow danger .. happens if shape is "......" 
        k -= 1
    #i and j are now the indices of the first unpaired bases
    # so we know that the first element we add is a stack .. 
    nextnod = len(G)
    G.add_node(nextnod,color = 'blue')
    G.node[nextnod]['type'] = 'Stack'
    G.node[nextnod]['abstract'] = True
    for x in range(start,i)+range(end,j,-1):
        G.add_edge(nextnod,x,color = 'pink',nesting = True)    
    unpaired = [] # number of unpaired bases in the loop/ml/hp/whatever 
    stemsfound = 0 # number of stems connected to us...
    asc = []     # abstract shape children // things we found in the revursion
    # we add these so the last bridge of a stack is part of the newly found ML or loop or...
    unpaired.append(i-1)
    unpaired.append(j+1)
    #we move along the unpaired bases... 
    while i <= j:
        if i == ie[k][0]: # => another stem begins.
                stemsfound += 1
                asc.append( _add_abstr(G,ie[k][0],ie[k][1],optlist) )
                unpaired.append(ie[k][0])
                unpaired.append(ie[k][1])
                # the recursion took care of everything behind that stem so we can just continue 
                # at the index of the next probably unpaired index
                i = ie[k][1]                
                #let the edge index catch up to us
                while ie[k][0] < i and k < len(ie)-1:
                        k += 1
        else:
            unpaired.append(i)
        i += 1    
    # lets add our results to the graph
    nextnode = len(G)
    #print unpaired,stemsfound
    typ = "hairpin"
    if stemsfound == 1:
        typ = "internal loop"
    if stemsfound >= 2:
        typ = "multiloop"
    G.add_node(nextnode,color = "red")
    G.node[nextnode]['type'] = typ
    G.node[nextnode]['abstract'] = True
    if optlist['connect_abstract_nodes']:
        G.add_edge(nextnod,nextnode,color = 'blue')
    
    for e in unpaired:
        G.add_edge(e,nextnode,nesting = True, color = optlist['color_loop_satelite'])
    
        
    if optlist['connect_abstract_nodes']:
        for cs in asc:
            G.add_edge(cs,nextnode,color = 'blue')
    return nextnod


def addbridges(structure,open ,close,G,info,optlist):
    '''
    if we find opening and closing bracket we make a respective 
    edge in the graph
    '''
    conop = []
    i = 0
    for e in structure:
        if e == open:
            conop.append(info+i+1)
        if e == close:
            G.add_edge(conop.pop(),info+i+1, view_point = True, color = optlist['color_bridge'] )
        i += 1


def has_structure(struct):
    '''
    we dont know if we have a structure for certain, so we better check
    '''
    return '(' in struct


def addgraph(optlist,data,G):
    #optlist['log'].debug("add to graph %s : %s" % (data.sequence_name , data.structure) )
    # create a central node for the sequence... and copy some attributes
    info_node = len(G)
    G.add_node(info_node)
    G.node[info_node].update(data.__dict__)
    G.node[info_node]['info_node'] = True
    
    # so we add all the nodes for the nucleoties
    for e in xrange(len(data.structure)):
        G.add_node(e+info_node+1)
        G.node[e+info_node+1]['label'] = data.sequence[e]
        G.node[e+info_node+1]['partial_order_index'] = e+int(data.start_id )

    # add all the bridges....
    addbridges(data.structure,'<','>',G,info_node,optlist)
    addbridges(data.structure,'[',']',G,info_node,optlist)
    addbridges(data.structure,'{','}',G,info_node,optlist)
    addbridges(data.structure,'(',')',G,info_node,optlist)

    # find main structure in the graph..
    # so for example i have ...(SOMETHING)..(SOMETING_ELSE).. i get the indices 
    # of the bracket characters. 
    # we do this before the backbone is added.. because then we dont need to filter backbone edges
    relevant = []
    if optlist['abstr'] and has_structure(data.structure):
        node = info_node+1 
        while node < len(G): # we go over all nodes
            ne = G.neighbors(node) 
            if len(ne) > 0: # iff we have neighbours
                ne = ne[0] # we can have only one. because we can only have a bridge with one other
                relevant.append((node,ne)) # and we add the brigde to 'relevant'
                node = ne+1 # this represents jumping over the SOMETHING , mentioned above.
            else:
                node += 1 # if we dint have neighbours we see if the next nucleotide has a neighbour...

    # connect backbone nodes...
    for i in range(len(data.sequence)-1):
        G.add_edge(info_node+i+1,info_node+i+2,view_point = True,color = optlist['color_backbone'])

    # stacks may get different kinds of special markers.. we do this here.
    _add_stacks(G,info_node,optlist)
    
    # if we have an annotationfile(specified in args ) we use it! 
    if optlist['annotate'] != None: # then: we set it to be a nice dictionary :) 
        if data.sequence_name in optlist['annotate']: # is there annotation?
            l = optlist['annotate'][data.sequence_name] 
            for (start, end , text) in l: # annotations are stretching over a range of nucleotides    
                for i in range( max( int(data.start_id), int(start)), min( int(end)+1, int(data.start_id)+len(data.sequence)) ): # and for each nucleotide
                    # i is now the real index in the sequence
                    n = len(G) # we add a node
                    G.add_node(n)
                    G.node[n]['label'] = text # and set some text
                    G.add_edge(n,info_node+i-int(data.start_id)+1 ) # and connect to the nucleotide
    
    if  optlist['abstr']:
        if has_structure(data.structure):
            # we established that relevant marks the beginning of probably a stack like this "...(SOMETHING)....(SOMETHING)..."            
            children = []
            for stack in relevant:
                children.append( _add_abstr(G,stack[0],stack[1],optlist) )  # so we find out what the SOMETHINGS are and find abstract structures

            # _add_abstr gave is a bunch of nodes that represent a stack.... we need to connect the abstract stack representations.
            for e in range(len(children)-1):
                if optlist['connect_abstract_nodes']:
                    G.add_edge( children[e], children[e+1], color = optlist['color_intra_abstract']) 

            #rest is dealing with dangling ends.. they also need to be connected to the SOMETHINGS...
            
            # first the right side
            r = range(0,find_opening(data.structure))
            nextnode = len(G)
            if len(r)>0:
                G.add_node(nextnode,color = 'green')
                G.node[nextnode]['gspanname'] = 'dangling'
                G.node[nextnode]['abstract'] = True
                if optlist['connect_abstract_nodes']: 
                    G.add_edge(nextnode,children[1],color = optlist['color_intra_abstract'])
                for e in r:
                    G.add_edge(e+info_node+1,nextnode,nesting = True,color = optlist['color_loop_satelite'])

            # then analog the left side
            l = range( find_closing(data.structure)+1,len(data.structure)) 
            nextnode = len(G)
            if len(l)>0:
                G.add_node(nextnode,color = 'green')
                G.node[nextnode]['gspanname'] = 'dangling'
                G.node[nextnode]['abstract'] = True
                if optlist['connect_abstract_nodes']:
                    G.add_edge(nextnode,children[-1],color = optlist['color_intra_abstract'])
                for e in l:
                    G.add_edge(e+info_node+1,nextnode,color = optlist['color_loop_satelite'],nesting = True)

    # if we are not in debug mode...  we connect the info_node with every node in the newly created subgraph.. 
    #this looks messy  when we draw the graph so we dont do it when in debug mode..
    if optlist["debug"] == False:
        for i in xrange(info_node+1,len(G)):
            G.add_edge( info_node,i,nesting = True )

    return info_node
    

def find_opening(string):
    ret = map(string.find,['<','(','{',"["])
    if -1 in ret:
        ret.remove(-1)
    return min(ret)


def find_closing(string):
    return max(map(string.rfind,['>',')','}',"]"]  ))


def create_super_graph(input = None, options = None):
    '''
    Takes in input a post_rna_struct object and outputs a networkx graph.

    Parameters
    ----------
    input : post_rna_struct object
        Post_rna_struct objects aee created by call_rna_'program_name'

    options : dict
        dictionary containing parameters
    '''

    G = nx.Graph()
    G.add_node(0)
    G.node[0].update(input.attributes)
    G.node[0]['info_node'] = True
    G.node[0]['supernode'] = True
    for e in input.sequence_groups:
        new_node = len(G)
        G.add_edge(0,new_node, nesting = True)
        # copy annotations to node. 
        G.node[new_node].update(e.attributes)

        G.node[new_node]['info_node'] = True
        for s in e.sequences:
            referencenode = addgraph(options,s,G)
            G.add_edge(new_node, referencenode, nesting = True)
    return G
