

def get_attribute_string(sth):
	return ' ; '.join([ key +' = '+sth[key] for key in sth ])

class post_rna_struct:
	'''
	this defines what actually comes out of an structure prediction programm wrapper function. 
	'''
	def __init__(self):
		self.attributes={}
		self.sequence_groups=[]

	def __str__(self):
		return get_attribute_string(self.attributes)

class sequence_group:
	def __init__(self):
		self.attributes={}
		self.sequences=[]

	def __str__(self):
		return get_attribute_string(self.attributes)


class sequence:
	def __init__(self):
		self.attributes={}
		self.start_id = -2
		self.sequencename=''
		self.seqence=''
		self.structure=''
		'''
		structure uses < { ( [ and . 
		'''
		
		
	def __str__(self):
		return get_attribute_string(self.__dict__)



