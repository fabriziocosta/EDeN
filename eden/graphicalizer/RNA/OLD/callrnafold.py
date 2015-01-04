import os
import subprocess as sp
#dummy object oO 
class Object(object):
	pass 



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


def call_rna_fold(optlist,seq,seqname):
	'''
	we want to create a dictionary that represents the sequence.
	here we see an example output


	also you will encounter 'gsSOMETING' often .. this refers to stuff that will later be put into to gspan file header.

	WE WANT TO CREATE THIS: 
	postRNAstructure # the class name
	-gsname (str)   #make dictionary  -> attributes  
	-windows [] //see below

	SEQUENCE_GROUP $ renames
	-annotation {}  # -> attributes.. data will be plugged into the networkx node that repreents the window later
	-gsname (str) # -> tabun remove ... and create on the fly at gs export
	-sequences [] // see below

	SEQUENCE
	-gsname (str) # -> attributes   
	-start_id  # introduce this.
	-sequencename str
	-seq  str
	-structure  ( str: allowed chars:  < { ( [ and . )


	'''
	
	

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
				cmd='echo "%s" | RNAfold --noPS' % seq  #optlist['mode']+" "+getAbstrLevel(optlist)+seq
				text=sp.check_output(cmd,shell=True)
				text=text.split('\n')[1:]
				
				sequence=Object()
				sequence.gsname='s # i am a sequence my length is %i and my energy is %s' % (len(seq),text[len(seq):] )
				sequence.sequencename=seqname
				sequence.seq=seq
				sequence.structure=text[0].split()[0]
				window=Object()
				res.windows.append(window)
				window.annotations={}
				window.gsname='w # i am a whole sequence window'
				window.sequences=[ sequence ]
				if optlist['cue']==True:
					cut_unpaired_ends(sequence)

				optlist['log'].debug('exec: '+cmd)


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
						window.annotations={}
						window.sequences=[ create_seq_graph(optlist, seq[pos:pos+windowSize] , \
									seqname+" starting at:"+str(pos+1)+"	windowsize in bp:"+windowSize )    ]
						res.windows.append(window)
						pos+=shift
			else:

				pos=0
				while pos+windowSize < len(seq):
					
					s=seq[pos:pos+windowSize]	
					cmd='echo "%s" | RNAfold --noPS' % s
					
					text=sp.check_output(cmd,shell=True)

					text=text.split('\n')[1:]

					sequence=Object()
					sequence.gsname='s # i am sequence %s my length is %i and my energy is %s' %  (s,len(seq),text[len(seq):] )
					sequence.sequencename=seqname
					sequence.seq=seq
					sequence.structure=text[0].split()[0]
					window=Object()
					res.windows.append(window)
					window.annotations={}
					window.gsname='w # i am a whole sequence window  windowsize=%i startpos=%i' % (windowSize,pos)
					window.sequences=[ sequence ]
					if optlist['cue']==True:
						cut_unpaired_ends(sequence)

					pos+=shift
					
					optlist['log'].debug('exec: '+cmd)

	return res
