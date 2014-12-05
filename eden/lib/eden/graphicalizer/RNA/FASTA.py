import networkx as nx

def FASTA_to_eden(input = None, input_type = None, options = dict()):
    """
    Takes a list of strings and yields networkx graphs.

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
    return _FASTA_to_eden(f)        
   

def string_to_networkx(line):
    G = nx.Graph()
    for id,character in enumerate(line):
        G.add_node(id, label = character, viewpoint = True)
        if id > 0:
            G.add_edge(id-1, id, label = '-', viewpoint = True)
    assert(len(G)>0),'ERROR: generated empty graph. Perhaps wrong format?'
    return G


def _FASTA_to_eden(data_str_list):
    line_buffer = ''
    for line in data_str_list:
        _line = line.strip().upper()
        if _line:
            if _line[0] == '>':
                if len(line_buffer) > 0:
                    yield string_to_networkx(line_buffer)
                line_buffer = ''
            else:
                line_buffer += _line
    if len(line_buffer) > 0:
        yield string_to_networkx(line_buffer)