#!/usr/bin/python
import sys
import subprocess as sp
import random
import os
import networkx as nx
import argparse
import matplotlib.pyplot as plt
from readfasta import read_fasta
from callrnashapes import call_rna_shapes
from callrnashapes import call_rna_shapes_cast
from createsupergraph import create_super_graph
from callrnafold import call_rna_fold
from draw import nx_draw_invisibility_support, nxdraw, nx_draw_dot
import logging
#dummy object oO
class Object(object):
	pass



# the object that the structure pred stuff gives us will be 
#  a class and we call it postRNAstucture

# check the draw invicible nodes stuff again.. also document... 

# create super graph function.. why is there g[0][subgraph] /???


# name for bases in the graph is LABEL 

#supergraph line 268ff annotate

# fix annoataion part in the addgraph function


# drawing in anderes file.
# output als alle unterstuetzten networkx teil nx_write ... 5 stueck aru
# argparse choices ..

# call shapes function  should not be able to mess with optlist.

# #################3
# super argparse version ..
# note to self: argparse is cancer
#####################

parser=argparse.ArgumentParser(description='fasta to gspan', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--export-format', choices=['write_adjlist','write_multiline_adjlist','write_edgelist',
	'write_weighted_edgelist',
	'write_gexf',
	'write_gml',
	'write_gpickle',
	'write_graphml',
	'write_yaml',
	'write_pajek',
	'write_shp',
	'none'], 
		default ='none',
		help='export format, other interesting options are -o and --export-filename')
parser.add_argument('--export-filename',default='out.txt',
		help='where to export the graph to.. see: --export-format')

parser.add_argument('-t',
		help='shapelevel 4=200=> winsize200 has level 4' )

parser.add_argument('-i',
		help='turn on samplemode with -i samples',default='0')

parser.add_argument('-r',
		help='calculate structure probabilities',
		action="store_true")

parser.add_argument('-c',
		help='energy range in kcal/mol')

parser.add_argument('-e',
		help='energy range')

parser.add_argument('-q',
		help='shape probability ',
		action='store_true')

parser.add_argument('-o',
		help='output dir',
		default="GSPAN/")

parser.add_argument('-M',
		help='max number of shrepf per window',
		type=int,
		default=-1)

parser.add_argument('--mode',
		help='pKiss or rnashapes or RNAfold',
		default='RNAshapes')
parser.add_argument('--fasta',
		help='path to a fasta file',
		required=True)

parser.add_argument('--wins',
		help='windowsize 20,40,50')

parser.add_argument('--shift',
		help='whift for windows in percent',
		type=int,
		default=1)

parser.add_argument('--cue',
		action='store_true',
		help='cut unpaired ends')

parser.add_argument('--stack',
		action='store_true',
		help='put P-node on stacks')

parser.add_argument('--Tp',
		help='filtecir cutoff for shape probs.. needs -q ')

parser.add_argument('--set-graph-win',
		action='store_true',
		help='add sequence graph to each window')

parser.add_argument('--set-graph-t',
		action='store_true',
		help='add sequence graph to each sequence')

parser.add_argument('--set-graph-alph',
		action='store_true',
		help='added sequence-graphs are named differently')

parser.add_argument('--abstr',
		action='store_true',
		help='add abstract graph and connect node...')

parser.add_argument('--debug',action='store_true',help='display more info')
parser.add_argument('--nostr',
		action='store_true',
		help='no structure')

parser.add_argument('--group',
		help='not implemented anymore oO',type=int,default=1 )

parser.add_argument('--stdout',
		action='store_true',
		help='write to console')

parser.add_argument('--cast',
		action='store_true',
		help='read all entries in fasta-file and create consensus shape..'+
			'the concept of window in the graph output shall be a little bit different here.. ')

parser.add_argument('--ignore_header',
		action='store_true',
		help='ignore header')

parser.add_argument('--annotate',
		help='annotationfile where?')

parser.add_argument('--match-shape',
		help='match this shape')

parser.add_argument('--vp',
		action='store_true',
		help='add viewpoint labels')





def prepare_optlist(optlist):



	# the plan is to log everything to file but only debug mode prints debug messages to console
	logger = logging.getLogger('loooog')
	logger.setLevel(logging.DEBUG)
	fh = logging.FileHandler('run.log')
	fh.setLevel(logging.DEBUG)

	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	if  optlist['debug']==False:
		ch.setLevel(logging.WARNING)

	formatter = logging.Formatter('%(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	fh.setFormatter(formatter)
	logger.addHandler(ch)
	logger.addHandler(fh)

	optlist['log']=logger




	optlist['color_backbone']='black'
	optlist['connect_abstract_nodes']=False
	optlist['color_intra_abstract']='red'
	optlist['color_bridge']='blue'
	optlist['color_loop_satelite']='pink'

	if optlist['debug']==True:
		optlist['debug_drawstyle']='grouped' # default, supporting_nodes , grouped
	else:
		optlist['debug_drawstyle']='default' # if it is supporting nodes things go bad.

	if not os.path.exists(optlist['o']):
			os.makedirs(optlist['o'])


	if optlist['wins']!=None:
		optlist['wins']= [int(x) for x in optlist['wins'].strip().split(',')]
		if 'shift' not in optlist:
			optlist['absolute_shift']=[1]
		else:
			i=int(optlist['shift'])
			optlist['absolute_shift']=[  max(1,e*i/100) for e in optlist['wins']]
	else:
		optlist['wins']=[-1]
		optlist['absolute_shift']=[0]



	if optlist['annotate']!=None:
		with open(optlist['annotate']) as f:
			t=f.read().split('\n')
			f.close()
			d={}
			for line in t:
				if len(line)>3:
					line=line.split(",")
					if line[0] not in d:
						d[line[0]]=[]
					d[line[0]].append([x.strip() for x in line [1:] ])
			optlist['annotate']=d



################################ MOVE


	#  optionen e c r i q und Tp were here.. they are now in the structure calculationmodule
####################################


# t= abstractionlevel .. 3=100,5=400 abstr level for different win sizes
#might not be the shortest way to write this, i am slightly drunk while writing this..
	optlist['default_abstraction_level']='3' # default abstraction level
	if optlist['mode']=='pKiss':
		optlist['default_abstraction_level']='-1'
	optlist['abstraction_level_dict']={} # abstractionlevel dictionary
	if optlist['t']!=None:
		s=optlist['t']
		if '=' in s:
			s=s.split(',')
			for e in s:
				level,wins=e.split('=')
				optlist['abstraction_level_dict'][wins]=level
		else:
			optlist['default_abstraction_level']=optlist['t']









def createGspan(optlist, G):
	'''
	# GSPANSTUFF.. print the 's # annotation' that should be G.annotation
	'''
	text=''
	edges=G.edges()
	edgeid=0
	for n in G.nodes():

		if 'gsname' in G.node[n]:
			while edges[edgeid][0]<n:
				text+="e "+str(edges[edgeid][0])+" "+str(edges[edgeid][1])+"\n"
				edgeid+=1
			text+=G.node[0]['gsname']+"\n"
		else:
			text+='v '+str(n)+ " "+G.node[n]['gspanname'] +"\n" #"+G.node[n]['label']+"\n"
	else:
		# add rest of edges
		text+="\n".join(["e "+str(e[0])+" "+str(e[1])  for e in edges[edgeid:] ])

	return text

def write(text,optlist,index):


	with open(optlist['o']+str(index)+".gspan","w") as f:
		f.write(text)
		f.close()


def debug(optlist,G):
		if optlist['debug']:
			if optlist['debug_drawstyle']=='supporting_nodes':
				nx_draw_invisibility_support(G)
			elif optlist['debug_drawstyle']=='default':
				nxdraw(G)
			elif optlist['debug_drawstyle']=='grouped':
				nx_draw_dot(G)
			exit()


if __name__=="__main__":


	#let argparse do its thing
	args=parser.parse_args()
	#we actually just want the dictionary...
	optlist=args.__dict__ 
	#to manipulate it:
	prepare_optlist(optlist)

	text=''

	# this is the consensus structure option, that will eat all the sequences at once
	if optlist['cast']:
		r=call_rna_shapes_cast(optlist)
		G=create_super_graph(r,optlist)
		debug(optlist,G)
		text=createGspan(optlist,G)

	else:
		for (sequencename,sequence) in read_fasta(optlist):
			
			if 'RNAfold' in optlist['mode']:
				r=call_rna_fold(optlist,sequence,sequencename)
			else:
				r=call_rna_shapes(optlist,sequence,sequencename) # second is the squence

			G=create_super_graph(r,optlist)
			debug(optlist,G)
			text+= createGspan(optlist,G)

	if optlist['stdout']:
		print text
	# export in any supported export_format
	elif optlist['export_format']!='none':
		function=eval('nx.'+optlist['export_format'])
		function(G,optlist['export_filename'])
	else:
		# just write gspan to a file
		write(text,optlist,'0')



#yield
#plt write
#isstructure



