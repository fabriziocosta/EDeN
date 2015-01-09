import networkx as nx
from rna_shapes import call_rna_shapes
from createsupergraph import create_super_graph
from rna_fold import call_rna_fold


def RNA_sequence_to_eden(input = None, input_type = None, options = dict()):
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
    return _RNA_sequence_to_eden(f, options = options)        
   

def _RNA_string_to_networkx(sequence = None, sequencename = None,  options = None):
    folding_mode = options['mode']
    if  folding_mode == 'RNAfold':
        r = call_rna_fold(sequence, sequencename, options)
    elif folding_mode == 'RNAshapes':
        r = call_rna_shapes(sequence, sequencename, options) # second is the squence
    else:
        raise Exception('Unknown structure predictor: %s' % folding_mode)
    G = create_super_graph(r,options)
    return G


def _RNA_sequence_to_eden(data_str_list, options = None):
    line_buffer = ''
    for line in data_str_list:
        _line = line.strip().upper()
        if _line:
            if _line[0] == '>':
                #extract string from header
                next_header_str = _line[1:] 
                if len(line_buffer) > 0:
                    G = _RNA_string_to_networkx(sequence = line_buffer, sequencename = header_str, options = options)
                    if options.get('header', False): 
                        G.graph['ID'] = header_str
                    yield G
                line_buffer = ''
                header_str = next_header_str
            else:
                line_buffer += _line
    if len(line_buffer) > 0:
        G = _RNA_string_to_networkx(sequence = line_buffer, sequencename = header_str, options = options)
        if options.get('header', False): 
            G.graph['ID'] = header_str
        yield G