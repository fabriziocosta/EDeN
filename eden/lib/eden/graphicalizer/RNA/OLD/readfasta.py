

def read_fasta(optlist):
	'''
	we dont build a dictionary but yield a sequence a time.
	'''
	sequencename=''
	sequence=''
	index=0
	with open(optlist['fasta']) as file:
		for line in file:
			if not line.strip():
				continue
			if line.startswith('>'):
				if sequencename:
					yield (sequencename,sequence)
				if optlist['ignore_header']:
					sequencename='seqnr: %s' % index
					index+=1
				else:
					sequencename=line.strip()[1:]
				sequence=''
			else:
				if optlist['vp']:
					sequence+=line.strip()
				else:
					sequence+=line.strip().upper()
		else:
			yield (sequencename,sequence)


