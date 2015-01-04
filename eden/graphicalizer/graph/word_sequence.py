import networkx as nx

def word_sequence_to_eden(input = None, input_type = None, options = dict()):
    """
    Takes a list of strings, splits each string in words and yields networkx graphs.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    input_type : ['url','file','string_file']
        If type is 'url' then 'input' is interpreted as a URL pointing to a file.
        If type is 'file' then 'input' is interpreted as a file name.
        If type is 'string_file' then 'input' is interpreted as a file name for a file 
        that contains strings rather than integers. The set of strings are mapped to 
        unique increasing integers and the corresponding vector of integers is returned.
    """
    input_types = ['url','file','list']
    assert(input_type in input_types),'ERROR: input_type must be one of %s ' % input_types

    if input_type == 'file':
        f = open(input,'r')
    elif input_type == 'url':
        import requests
        f = requests.get(input).text.split('\n')
    elif input_type == "list":
        f = input
    return _word_sequence_to_eden(f)        
   

def word_sequence_to_networkx(line):
    G = nx.Graph()
    for id,token in enumerate(unicode(line, errors = 'replace').split()):
        G.add_node(id, label = token, viewpoint = True)
        if id > 0:
            G.add_edge(id-1, id, label = '-')
    assert(len(G)>0),'ERROR: generated empty graph. Perhaps wrong format?'
    return G


def _word_sequence_to_eden(data_list):
    for word_sequence in data_list:
        yield word_sequence_to_networkx(word_sequence)