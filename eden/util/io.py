import numpy as np


def load_target(name, input_type='url'):
	"""
    Return a numpy array of integers to be used as target vector.

    Parameters
    ----------
    name : string
        A pointer to the data source.

    input_type : ['url','file','string_file']
        If type is 'url' then 'name' is interpreted as a URL pointing to a file.
        If type is 'file' then 'name' is interpreted as a file name.
        If type is 'string_file' then 'name' is interpreted as a file name for a file 
        that contains strings rather than integers. The set of strings are mapped to 
        unique increasing integers and the corresponding vector of integers is returned.
    """
    
	input_types = ['url','file','string_file']
	assert(input_type in input_types),'ERROR: input_type must be one of %s ' % input_types
	
	if input_type is 'file':
		with open(name,'r') as f:
			Y=[y.strip() for y in f]
	    	return np.array(Y,int)
	elif input_type is 'url':
		import requests
		r = requests.get(name)
		Y=[y for y in r.text.split()]
		return np.array(Y,int)
	elif input_type is 'string_file':
		with open(name,'r') as f:
			Ys=[y.strip() for y in f]
			target_dict=set(Ys)
			target_map={}
			for id,name in enumerate(target_dict):
				target_map[name]=id
			Y=[target_map[target] for target in Ys]
			return np.array(Y,int)
	else:
		raise Exception('Unidentified input_type:%s'%input_type)