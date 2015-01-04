import os
import subprocess as sp
from post_rna_struct import post_rna_struct,sequence_group, sequence 




def getAbstrLevel(optlist,wins=''):
	if str(wins) in optlist['abstraction_level_dict']:
		return " --shapeLevel "+optlist['abstraction_level_dict'][str(wins)]+" "
	elif int(optlist['default_abstraction_level'])>0:
		return " --shapeLevel "+optlist['default_abstraction_level']+" "
	return ' ' 


def cut_unpaired_ends(o):
	s=o.structure.find("(")
	e=o.structure.rfind(")")+1
	o.structure=o.structure[s:e]
	o.sequence=o.sequence[s:e]

def create_seq_graph(optlist,seq,name, sid=0):
	'''
		will create a dummy graph without structure.. 
		... the plan is to pretend that this dummy is rnashape output so we
		... the graphbuilder will just build a normal graph..
	'''
	data=sequence()
	if False==optlist['set_graph_alph']:
		data.sequence=seq
	else:
		data.sequence=seq.lower()
	data.sequencename=name
	data.start_id=sid
	data.structure="."*len(seq)
	data.attributes['info']=" you wanted something without structure so we deliver"
	return data 

def make_annotation(optlist,shapestuff,sequencestuff,o):
	'''
	just creates the annotations above the actual gspan graphs... 
	'''
	o.attributes['end']=sequencestuff[2]
	o.attributes['shape']=shapestuff[-1]
	o.attributes['energy']=shapestuff[0]
	if optlist['r']:
		o.attributes['probability']= shapestuff[1]




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
	
	text=sp.check_output(cmd,shell=True)
	if 'No consensus shapes found' in text:
		optlist['log'].critical("no consensus shapes found")
		return post_rna_struct()
	text=text.split('\n')
	i=0
	

	ret=post_rna_struct()
	currentwindow=sequence_group()
	while i < len(text):
		if not text[i]:
			i+=1
			continue
		if text[i][0] != '>' and 'Shape' in text[i]:
			window=sequence_group()
			ret.sequence_groups.append(window)
			currentwindow=window
			i+=1

		if text[i][0] == '>':
			gsname=text[i]+text[i+2].split()[0]
			
			shape=text[i+2].split()[1]
			sequenc=sequence()
			sequenc.attributes['info']='s # %s' % gsname			
			sequenc.sequencename=text[i][1:]
			sequenc.sequence=text[i+1].split()[1]
			sequenc.start_id=1
			sequenc.structure=text[i+2].split()[1]
			currentwindow.sequences.append(sequenc)

			i+=3
	return ret


def call_rna_shapes(optionlist=None,seq=None,seqname=None):
	'''
	we create a post_rna_struct by calling rna_shapes and filling
	the structure.. 
	'''
	optlist=optionlist.copy() # we dont want to mess up the original
	do_parameter_transformation(optlist)

	res=post_rna_struct()
	if optlist['set_graph_t']:
		tmp=sequence_group()
		tmp.sequences=[createSeqGraph(optlist,seq,seqname) ]
		tmp.attributes={'beginning':0,"end":len(seq) }
		res.sequence_groups.append(tmp)
	

	text=[]

	for (windowSize,shift) in zip(optlist['wins'],optlist['absolute_shift']):
		if windowSize== -1:
			if  optlist['nostr']==False:
				cmd=optlist['path_to_program']+" "+getAbstrLevel(optlist)+optlist['rnashape_parameter_string']+seq
				
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
						window=sequence_group()
						window.attributes['info']='w # you used the nostr option. here be dragons  '
						window.sequences=[ create_seq_graph(optlist, seq[pos:pos+windowSize] , \
									seqname+" starting at:"+str(pos+1)+"	windowsize in bp:"+windowSize ,sid=pos+1)    ]
						res.sequence_groups.append(window)
						pos+=shift
			else:
				#first line of output is just a hint on the sequencename -> ignore.
				cmd=optlist['path_to_program']+" --windowSize "+str(windowSize)+" --windowIncrement "+str(shift)+getAbstrLevel(optlist,windowSize)+optlist['rnashape_parameter_string']+" "+seq
				
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
			
			window=sequence_group()
			res.sequence_groups.append(window)
			window.attributes={'info':line,'beginning':sequenceline[0],'end':sequenceline[2]}

			currentwindow=window

		elif nex=='data':
			if maxshapecounter == optlist['M']:
				continue
			line=line.split() # [energy, shape, abstractshape]
			if optlist['match_shape']!=None and line[-1]!=optlist['match_shape']:
				continue
			maxshapecounter+=1

			sequenceobject=sequence()
			sequenceobject.start_id=sequenceline[0]	
				
			#print "asdasd  "+ str(sequenceline)
			sequenceobject.sequence=sequenceline[1]	
			sequenceobject.structure=line[1+optlist['r']] #kekeke
			
			if optlist['cue']:
				cut_unpaired_ends(sequenceobject)
			make_annotation( optlist,line,sequenceline,sequenceobject)
			sequenceobject.sequencename=seqname

			currentwindow.sequences.append(sequenceobject)

	return res
