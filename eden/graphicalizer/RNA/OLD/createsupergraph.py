import networkx as nx

import subprocess as sp
'''
ok here is how this section works .. 
create_super_graph will eat a speciffic object that can be produced eg by callrnashapes.. 
it will create an networkx graph.

the structure data that we need looks like this: 

	THING
	-gsname (str)
	-windows [] //see below

	WINDOW
	-annotation {}  .. data will be plugged into the networkx node that repreents the window later
	-gsname (str)
	-sequences [] // see below

	SEQUENCE
	-gsname (str)
	-sequencename str
	-seq  str
	-structure  ( str: allowed chars:  < { ( [ and . )


'''




def addstacks(G,start,optlist): 
	'''
	we put supporting 'p' nodes in stacks...
	'''

	#interesting edges:
	ie = [  a  for a in G.edges() if a[0]-a[1] != -1 and a[0] > start ]
	nextnode=len(G)
	#good thing that edges are sorted,..,.
	csum=0
	for (a,b) in ie:
		if a+b==csum:
			#found a stacking oO


			# ok lets add invisible supporting nodes that make the drawing more pretty
			if optlist['debug_drawstyle']=='supporting_nodes':
				G.add_node(nextnode)
				G.node[nextnode]['style']='invisible'
				for i in [a-1,a,b,b+1]:
					G.add_edge(i,nextnode,style='invisible')
				nextnode+=1



			# actually add stacks..  
			if optlist['stack']:
				G.add_node(nextnode)
				G.node[nextnode]['gspanname']='P'
				for i in [a-1,a,b,b+1]:
					G.add_edge(i,nextnode)
				nextnode+=1

		else:
			csum=a+b
	#return G



############################
# start end end mark the beginning of a stem.. we need to check that one out... 
# 
#########################
def addabstr(G,start,end,optlist):
	'''
	we add some nodes that indicate the structure element a base belongs to.



	1. most often  start and end will mark the beginning of a stacking.. so we walk along the stacking first. 
		 
	2. then we walk along the unpaired bases until we find an other stacking in which case: recurse. 

	.. in the end we find out what this structure actually is by seing how many times we found other stackings.

	we also connect ehis elements with the children found.. '''
	ie = [  e  for e in G.edges() if e[0]-e[1] != -1 and e[0]>start and e[1]<=end] 


	# we know that start binds to end so we do ths:
	i=start+1
	j=end-1

	k=0 # k is the index in ie
	for (b,e) in ie:
		if (b,e) != (i,j):
			break
		else:
			i+=1
			j-=1
		k+=1
	
	if k==len(ie):#overflow danger .. happens if shape is "......" 
		k-=1
	
	
	nextnod=len(G)
	G.add_node(nextnod,color='blue')
	G.node[nextnod]['gspanname']='Stack'
	G.node[nextnod]['abstract']=True

	for x in range(start,i)+range(end,j,-1):
		G.add_edge(nextnod,x,color='pink',nesting=True)
		

	
	# these two are basically what we want to know
	unpaired=[] 
	stemsfound=0
	asc=[]	 # abstract shape children
	unpaired.append(i-1)
	unpaired.append(j+1)

	while i <= j:
		#print "index="+str(i)
		if i == ie[k][0]: # => another stem begins.
				stemsfound+=1
				asc.append( addabstr(G,ie[k][0],ie[k][1],optlist) )
				unpaired.append(ie[k][0])
				unpaired.append(ie[k][1])
				i=ie[k][1]
			
				while ie[k][0]<i and k < len(ie)-1:
						k+=1
		else:
			unpaired.append(i)
		i+=1
	# lets add our results to the graph
	
	nextnode=len(G)
	#print unpaired,stemsfound
	typ="hairpin"
	if stemsfound == 1:
		typ="internal loop"
	if stemsfound >=2:
		typ="multiloop"
	
	G.add_node(nextnode,color="red")

	G.node[nextnode]['gspanname']=typ
	G.node[nextnode]['abstract']=True
	
	if optlist['connect_abstract_nodes']:
		G.add_edge(nextnod,nextnode,color='blue')
	
	for e in unpaired:
		G.add_edge(e,nextnode,nesting=True, color=optlist['color_loop_satelite'])
	
	if optlist['debug_drawstyle']=='supporting_nodes':
		newnode=len(G)
		G.add_node(newnode)
		G.node[newnode]['style']='invisible'
		G.node[newnode]['size']='500'
		for i in unpaired:
			G.add_edge(i,newnode,style='invisible')
	if optlist['connect_abstract_nodes']:
		for cs in asc:
			G.add_edge(cs,nextnode,color='blue')
	return nextnod




def addbridges(structure,open ,close,G,info,optlist):

	conop=[]
	i=0
	for e in structure:
		if e == open:
			conop.append(info+i+1)
		if e == close:
			G.add_edge(conop.pop(),info+i+1, view_point=True,color= optlist['color_bridge'] )#style='invis')
		i+=1

def has_structure(struct):
	return '(' in struct

def addgraph(optlist,data,G):

	optlist['log'].debug("adding"+data.structure)
	#if optlist['debug']:
	#	print "addgraph: "+data.structure	
		
	
	#G=nx.Graph()
	#G.rnashape=data.shape # its nice for debugging graphs to have this available.

	
	info_node=len(G)
	G.add_node(info_node)

	G.node[info_node]['sequence']=data.seq
	G.node[info_node]['annotation']=data.gsname
	G.node[info_node]['gsname']=data.gsname
	G.node[info_node]['structure']=data.structure
	G.node[info_node]['sequencename']=data.sequencename

	#G.add_nodes_from(xrange(info_node+1,info_node+1+len(data.shape)))
	for e in xrange(len(data.structure)):
		G.add_node(e+info_node+1)
		#G.node[e+info_node+1]['lable']=str(e)  
		G.node[e+info_node+1]['gspanname']=data.seq[e]

	

	# add all the stacks.. 
	addbridges(data.structure,'<','>',G,info_node,optlist)
	addbridges(data.structure,'[',']',G,info_node,optlist)
	addbridges(data.structure,'{','}',G,info_node,optlist)
	addbridges(data.structure,'(',')',G,info_node,optlist)


	# find main structure in the graph..

	# so example i have ...(SOMETHING)..(SOMETJINGELSE).. i get the indices 
	# of the bracket characters. 

	# we do this before the backbone is added.. because then we dont need to filter backbone edges
	relevant=[]
	if  optlist['abstr'] and  "("  in data['structure']:
		# we just ask for ( to see if there is a structure.
		# the actual search is in the graph already oO
		node=info_node+1 
		while node < len(G):
			ne=G.neighbors(node) 
			if len(ne) > 0:
				ne=ne[0] # there can be only one.. so we unpack oO
				relevant.append((node,ne))
				node=ne+1
			else:
				node+=1



	# connect backbone nodes...
	for i in range(len(data.seq)-1):
		G.add_edge(info_node+i+1,info_node+i+2,view_point=True,color=optlist['color_backbone'])#style='invis')


	addstacks(G,info_node,optlist)
	


	# if we have an annotationfile(specified in args ) we use it! 
	if optlist['annotate']!=None: # then: we set it to be a nice dictionary :) 
		if data.sequencename in optlist['annotate']:
			l=optlist['annotate'][data.sequencename]
			for (start, end , text) in l:
				for i in range(int(start),int( end)+1):
					n=len(G) 
					G.add_node(n)
					G.node[n]['gspanname']=text
					G.add_edge(n,info_node+i )
						
			
		
	if  optlist['abstr']:
		if has_structure(data.structure):
		#if "("  in data.structure:

			####
			# ok lets first find the relevant structures... 
			####
			
			children=[]
			for stack in relevant:
				children.append( addabstr(G,stack[0],stack[1],optlist) ) 
			
			for e in range(len(children)-1):

				if optlist['connect_abstract_nodes']:
					G.add_edge( children[e], children[e+1], color=optlist['color_intra_abstract']) 

			#rest is dealing with dangling ends
			r=range(0,find_opening(data.structure))

			nextnode=len(G)
			if len(r)>0:
				G.add_node(nextnode,color='green')
				G.node[nextnode]['gspanname']='dangling'
				G.node[nextnode]['abstract']=True
				if optlist['connect_abstract_nodes']: 
					G.add_edge(nextnode,children[1],color=optlist['color_intra_abstract'])
				for e in r:
					G.add_edge(e+info_node+1,nextnode,nesting=True,color=optlist['color_loop_satelite'])

			l=range( find_closing(data.structure)+1,len(data.structure))
			nextnode=len(G)
			if len(l)>0:
				G.add_node(nextnode,color='green')
				G.node[nextnode]['gspanname']='dangling'
				G.node[nextnode]['abstract']=True
				if optlist['connect_abstract_nodes']:
					G.add_edge(nextnode,children[-1],color=optlist['color_intra_abstract'])
				for e in l:
					G.add_edge(e+info_node+1,nextnode,color=optlist['color_loop_satelite'],nesting=True)



	if optlist["debug"]==False:
		for i in xrange(info_node+1,len(G)):
			G.add_edge( info_node,i,nesting=True )

		
	return info_node
	


def find_opening(string):
	ret=map(string.find,['<','(','{',"["])
	if -1 in ret:
		ret.remove(-1)
	return min(ret)
def find_closing(string):
	return max(map(string.rfind,['>',')','}',"]"]  ))


def create_super_graph(data,optlist):
	G=nx.Graph()
	G.add_node(0)
	
	G.node[0]['subgraphs']=[]

	G.node[0]['gsname']=data.gsname


	for e in data.windows:
		newnode=len(G)
		G.add_edge(0,newnode,nesting=True)
		# copy annotations to node. 
		G.node[newnode].update(e.annotations)

		G.node[newnode]['gsname']=e.gsname
		for s in e.sequences:
			referencenode=addgraph(optlist,s,G)
			G.add_edge(newnode,referencenode,nesting=True)
	return G



