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
        If type is 'list' then 'input' is interpreted as a list of strings.
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
    return _FASTA_to_eden(f, options = options)        
   

def string_to_networkx(line, options = None):
    G = nx.Graph()
    if options.viewpoints == False:
        for id,character in enumerate(line):
            G.add_node(id, label = character)
            if id > 0:
                G.add_edge(id-1, id, label = '-')
    else:
        for id,character in enumerate(line):
            if character in 'AUGC':
                G.add_node(id, label = character, weight = 1)
            else:
                G.add_node(id, label = character, weight = 0.1)
            if id > 0:
                G.add_edge(id-1, id, label = '-')
    assert(len(G)>0),'ERROR: generated empty graph. Perhaps wrong format?'
    return G


def _FASTA_to_eden(data_str_list, options = None):
    line_buffer = ''
    for line in data_str_list:
        _line = line.strip().upper()
        if _line:
            if _line[0] == '>':
                #extract string from header
                header_str = _line[1:] 
                if len(line_buffer) > 0:
                    G = string_to_networkx(line_buffer, options = options)
                    G.graph['ID'] = prev_header_str
                    yield G
                line_buffer = ''
                prev_header_str = header_str
            else:
                line_buffer += _line
    if len(line_buffer) > 0:
        G = string_to_networkx(line_buffer, options = options)
        G.graph['ID'] = prev_header_str
        yield G