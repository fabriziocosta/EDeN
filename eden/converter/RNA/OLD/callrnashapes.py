import os
import subprocess as sp
#dummy object oO 
class Object(object):
	pass 


def getAbstrLevel(optlist,wins=''):
	if str(wins) in optlist['abstraction_level_dict']:
		return " --shapeLevel "+optlist['abstraction_level_dict'][str(wins)]+" "
	elif int(optlist['default_abstraction_level'])>0:
		return " --shapeLevel "+optlist['default_abstraction_level']+" "
	return ' ' 


def cut_unpaired_ends(o):
	s=o.structure.find("(")
	e=o.structure.rfind(")")+1
	# i assume that these are aleays found.. if not then there is no structure and tabun no output by rnashape
	o.structure=o.structure[s:e]
	o.seq=o.seq[s:e]

def create_seq_graph(optlist,seq,name):
	'''
		will create a dummy graph without structure.. 
		... the plan is to pretend that this dummy is rnashape output so we
		... the graphbuilder will just build a normal graph..
	'''
	data=Object()
	#if '--seq-graph-alph' not in optlist[]:
	if False==optlist['set_graph_alph']:
		data.seq=seq
	else:
		data.seq=seq.lower()#["s"+e for e in seq]
	data.sequencename=name
	data.structure="."*len(seq)
	data.gsname="s # "+name+" no structure."
	return data 

def make_annotation(optlist,shapestuff,sequencestuff,name):
	'''
	just creates the annotations above the actual gspan graphs... 
	'''
	return name+" energy,"+("prob," if optlist['r'] else "" )+\
		"shape,shape="+" ".join(shapestuff)+" start,seq,end="+" ".join(sequencestuff)




def do_parameter_transformation(optlist):
	# setting parameters for callrnashapes. 
	optlist['rnashape_parameter_list']=[]
	if optlist['e']!=None:
		optlist['rnashape_parameter_list']+=['--absoluteDeviation',optlist['e']] #rnashapeparapeters -> rnashape_parameter_list



	if  optlist['c']!=None:
		if '--absoluteDeviation' in optlist['rnashape_parameter_list']:
			print "-e and -c are mutually exclusive"
			exit()
		else: 
			optlist['rnashape_parameter_list']+=['--relativeDeviation',optlist['c']]


	if optlist['r']:
		optlist['rnashape_parameter_list']+=['--structureProbs 1']


	if optlist['i'] != '0':
			optlist['rnashape_parameter_list']+=['--mode sample --numSamples '+optlist['i']]

	if optlist['q']:
			optlist['rnashape_parameter_list']+=['--mode probs']

	if optlist['Tp']!=None:
		optlist['rnashape_parameter_list']+=['--outputLowProbFilter',optlist['Tp']]

	optlist['rnashape_parameter_string']=' '+" ".join(optlist['rnashape_parameter_list'])+" "


def call_rna_shapes_cast(optlist):
	do_parameter_transformation(optlist)
	cmd='RNAshapes -mode cast %s %s' % (optlist['rnashape_parameter_string'] ,optlist['fasta'] )
	if optlist['wins'][0] !=  -1:
		oplist['log'].warn('rnashapes cast does not support windowing option')
	if optlist['set_graph_win']:
		optlist['log'].warn('set_graph_win not implemented for cast ..')
	
	text=sp.check_output(cmd,shell=True).split('\n')
	i=0
	

	ret=Object()
	ret.gsname='t # i am a rna "cast" my concept of windowing might be weired'
	ret.windows=[]
	currentwindow=Object()
	while i < len(text):
		if not text[i]:
			i+=1
			continue
		if text[i][0] != '>' and 'Shape' in text[i]:
			window=Object()
			window.annotations={}
			window.gsname='w # %s' % text[i]
			window.sequences=[]
			ret.windows.append(window)
			currentwindow=window
			i+=1

		if text[i][0] == '>':
			gsname=text[i]+text[i+2].split()[0]
			
			shape=text[i+2].split()[1]
			sequence=Object()
			sequence.gsname='s # %s' % gsname			
			sequence.sequencename=text[i][1:]
			sequence.seq=text[i+1].split()[1]
			sequence.structure=text[i+2].split()[1]
			currentwindow.sequences.append(sequence)

			i+=3
	return ret


def call_rna_shapes(optlist,seq,seqname):
	'''
	we want to create a dictionary that represents the sequence.
	here we see an example output


	also you will encounter 'gsSOMETING' often .. this refers to stuff that will later be put into to gspan file header.

	WE WANT TO CREATE THIS: 
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
	do_parameter_transformation(optlist)

	res=Object()
	res.gsname='t # '
	res.windows=[]
	if optlist['set_graph_t']:
		tmp=Object()
		tmp.sequences=[createSeqGraph(optlist,seq,seqname) ]
		tmp.annotations={'beginning':0,"end":len(seq) }
		tmp.gsname='w # sequence blub'
		res.windows.append(tmp)
	

	text=[]
	#print optlist['wins']
	#print optlist['absolute_shift']
	for (windowSize,shift) in zip(optlist['wins'],optlist['absolute_shift']):
		if windowSize== -1:
			if  optlist['nostr']==False:
				cmd=optlist['mode']+" "+getAbstrLevel(optlist)+optlist['rnashape_parameter_string']+seq
				
				text=sp.check_output(cmd,shell=True).split('\n')[1:]
				optlist['log'].debug(cmd)




		else:
			if  optlist['nostr']:
				if optlist['set_graph_win']:
					pos=0
					while pos+windowSize < len(seq):
						'''
						create fake window with empty structure...
						'''	
						window=Object()
						window.gsname='w # you used the nostr option. here be dragons  '
						window.sequences=[ create_seq_graph(optlist, seq[pos:pos+windowSize] , \
									seqname+" starting at:"+str(pos+1)+"	windowsize in bp:"+windowSize )    ]
						res.windows.append(window)
						pos+=shift
			else:
				#first line of output is just a hint on the sequencename -> ignore.
				cmd=optlist['mode']+" --windowSize "+str(windowSize)+" --windowIncrement "+str(shift)+getAbstrLevel(optlist,windowSize)+optlist['rnashape_parameter_string']+" "+seq
				#text+=os.popen(cmd).read().split('\n')[1:]
				text+=sp.check_output(cmd,shell=True).split('\n')[1:]
				optlist['log'].debug(cmd)
	nex='seq'
	sequenceline=()
	maxshapecounter=0 # counter for the -M option... 
	currentwindow={}
	for line in text:
		line=line.strip()
		if len(line)==0:
			nex='seq'
			continue
		elif nex=='seq':
			
			sequenceline=line.split()
			nex='data'
			maxshapecounter=0
			#if  optlist['set_graph_win']:
			#	res.append(createSeqGraph(optlist,sequenceline[1],sequenceline[0]),seqname)
			
			window=Object()
			res.windows.append(window)
			window.gsname="w # "+line	
			window.annotations={'beginning':sequenceline[0],'end':sequenceline[2]}

			window.sequences=[]

			currentwindow=window

		elif nex=='data':
			if maxshapecounter == optlist['M']:
				continue
			line=line.split() # [energy, shape, abstractshape]
			if optlist['match_shape']!=None and line[-1]!=optlist['match_shape']:
				continue
			maxshapecounter+=1

			sequenceobject=Object()
			
			sequenceobject.seq=sequenceline[1]	
			sequenceobject.structure=line[1+optlist['r']] #kekeke
			#### i dont know what cue does to the annotation.. so it does nothing for now.
			if optlist['cue']:
				cut_unpaired_ends(sequenceobject)
			sequenceobject.gsname=make_annotation( optlist,line,sequenceline," # "+seqname)
			sequenceobject.sequencename=seqname

			currentwindow.sequences.append(sequenceobject)

	return res
