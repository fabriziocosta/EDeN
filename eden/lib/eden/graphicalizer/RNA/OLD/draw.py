#draw.py

import subprocess as sp
import networkx as nx
def nx_draw_invisibility_support(G):
	#A.overlap='scale'
	for e in G.edges():
		G[e[0]][e[1]]['len']=0.5

	for n in G.nodes():
		G.node[n]['shape']='circle'
		G.node[n]['width']='0.5'
		G.node[n]['height']='0.5'
		G.node[n]['size']='0.5'
		G.node[n]['label']='Z'

	A = nx.to_agraph(G)

	#A.layout()
	#A.draw('color.png')
	A.write('test.dot')
	sp.call('display color.png',shell=True)


def nxdraw(G):
	'''
	just printing and showing a graph...
	'''
	label=[ (n,G.node[n].get('gsname',G.node[n].get('gspanname','noname')))  for n in G.nodes()   ]
	label=dict(label)
	ncolor=[ G.node[n].get('color','white')  for n in G.nodes()   ]

	# this marks nesting edges... normaly nesting edges are disabled in debug mode..
	#echooser= lambda x: 'pink' if x==True else "black"
	#ecolor=[ echooser ('nesting' in e[2])  for e in G.edges(data=True) ]

	ecolor=[ e[2].get('color','black')   for e in G.edges(data=True)]

	nx.draw_graphviz(G,prog='neato',with_labels=True,
			labels=label,
			node_color=ncolor,
			edge_color=ecolor)
	plt.show()
	nx.write_dot(G,'multi.dot')


def get_nests(G,node,done,answer):
	''' belongs so nx_draw_dot'''
	# we want the children that are not done yet.
	children = [x for x in nx.all_neighbors(G,node) if x not in done]
	children = [x for x in children if 'nesting' in G[node][x]]
	#if we are end-nodes we dont do anything
	#end nodes are recognizeable because they have only abstract children...
	for c in children:
		if 'abstract' not in G.node[c]:
			break
	else:
		return
	#we dont need to do this node gain..
	done.append(node)
	# add to result and recurse
	if len(children)>0:
		answer[node]=children
		for c in children:
			get_nests(G,c,done,answer)


def get_subgraph_text(key,nests,done):
	''' belongs so nx_draw_dot'''
	if key in done:
		return ''
	done[key]=''
	if key not in nests:
		return "%s;\n" % key
	childs= nests[key]
	text='\nsubgraph cluster_'+str(key)+"{\n"+str(key)+";\n"

	f=lambda x: 0 if x in nests else 1
	childs=sorted(childs,key=f)
	for child in childs:
			text+= get_subgraph_text(child,nests,done)
	text+='}\n'
	return text



def nx_draw_dot(G):
	nests={}
	get_nests(G,0,[],nests)
	text=get_subgraph_text(0,nests,{})


	e=[ e for e in  G.edges(data=True) if 'nesting' in e[2]]
	G.remove_edges_from(e)


	nx.write_dot(G,'multi.dot')



	#os.popen('neato -Goverlap=false -T png  multi.dot > multi.png')
	sp.call("head -n -1 multi.dot > tmp.dot; echo \""+text+"}\n\">> tmp.dot",shell=True)
	sp.call('fdp -T png  tmp.dot > multi.png',shell=True)
	sp.call('display multi.png',shell=True)

