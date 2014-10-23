import networkx as nx

def sequence_to_eden(name, input_type = 'url'):
    """
    Takes a list of strings and yields networkx graphs.

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

    input_types = ['url','file','list']
    assert(input_type in input_types),'ERROR: input_type must be one of %s ' % input_types

    if input_type is 'file':
        f = open(name,'r')
    elif input_type is 'url':
        import requests
        f = requests.get(name).text.split('\n')
    elif input_type == "list":
        f = name
    return _sequence_to_eden(f)        
   

def _sequence_to_eden(data_str_list):
    def sequence_to_networkx(line):
        G = nx.Graph()
        for id,character in enumerate(line):
            G.add_node(id, label = character, viewpoint = True)
            if id > 0:
                G.add_edge(id-1, id, label = '-', viewpoint = True)
        assert(len(G)>0),'ERROR: generated empty graph. Perhaps wrong format?'
        return G

    string_list = []
    for line in data_str_list:
        if line.strip():
            yield sequence_to_networkx(line.strip())