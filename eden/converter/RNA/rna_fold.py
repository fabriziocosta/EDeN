import os
import subprocess as sp
from post_rna_struct import post_rna_struct, sequence_group, sequence 



def cut_unpaired_ends(o):
	s = o.structure.find("(")
	e = o.structure.rfind(")")+1
	o.structure = o.structure[s:e]
	o.sequence = o.sequence[s:e]

def create_seq_graph(local_options,seq,name):
	'''
		will create a dummy graph without structure.. 
		... the plan is to pretend that this dummy is rnashape output so we
		... the graphbuilder will just build a normal graph..
	'''
	data = sequence()
	if 	local_options.get('set_graph_alph', False) == False:
		data.sequence = seq
	else:
		data.sequence = seq.lower()#["s"+e for e in seq]
	data.sequencename = name
	data.structure = "."*len(seq)
	data.attributes['info'] =  'no structure created'
	return data 


def rna_fold_wrapper(seq  =  None, seqname  =  None, options  =  None):
	'''
	see the post_rna_struct class for infos on what we create here.
	'''
	local_options = dict(options) # we dont want to mess up the original
	res = post_rna_struct()
	if 	local_options.get('set_graph_t', False):
		tmp = sequence_group()
		tmp.sequences = [createSeqGraph(local_options,seq,seqname) ]
		tmp.annotations = {'beginning':0,"end":len(seq) }
		res.sequence_groups.append(tmp)
	text = []
	#print local_options['wins']
	#print local_options['absolute_shift']
	for (windowSize,shift) in zip(local_options['wins'],local_options['absolute_shift']):
		if windowSize ==  -1:
			if 	local_options.get('nostr', False) == False:
				cmd = 'echo "%s" | %s --noPS' % (seq,local_options['path_to_program'])  #local_options['mode']+" "+getAbstrLevel(local_options)+seq
				text = sp.check_output(cmd,shell = True)
				text = text.split('\n')[1:]
				sequen = sequence()
				sequen.sequencename = seqname
				sequen.sequence = seq
				sequen.structure = text[0].split()[0]
				window = sequence_group()
				res.sequence_groups.append(window)
				window.sequences = [ sequen ]
				if local_options.get('cue', False) == True:
					cut_unpaired_ends(sequen)
				#local_options['log'].debug('exec: '+cmd)
		else:
			if local_options.get('nostr', False):
				if local_options.get('set_graph_win', False):
					pos = 0
					while pos+windowSize < len(seq):
						'''
						create fake window with empty structure...
						'''	
						window = sequence_group()
						window.annotations['gsname'] = 'w # you used the nostr option. here be dragons  '
						window.sequences = [ create_seq_graph(local_options, seq[pos:pos+windowSize] , \
									seqname+" starting at:"+str(pos+1)+"	windowsize in bp:"+windowSize )    ]
						res.sequence_groups.append(window)
						pos += shift
			else:
				pos = 0
				while pos+windowSize < len(seq):
					s = seq[pos:pos+windowSize]	
					cmd = 'echo "%s" | RNAfold --noPS' % s
					text = sp.check_output(cmd,shell = True)
					text = text.split('\n')[1:]
					sequen = sequence()
					sequen.sequencename = seqname
					sequen.sequence = seq
					sequen.structure = text[0].split()[0]
					window = sequence_group()
					res.sequence_groups.append(window)
					window.attributes['gsname'] = 'w # i am a whole sequence window  windowsize = %i startpos = %i' % (windowSize,pos)
					window.sequences = [ sequen ]
					if local_options.get('cue', False) == True:
						cut_unpaired_ends(sequen)
					pos += shift
					#local_options['log'].debug('exec: '+cmd)
	return res