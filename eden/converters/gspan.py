import json
import networkx as nx

def gspan_to_eden(name, input_type='url'):
    """
    Takes a string list in the extended gSpan format and yields networkx graphs.

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
        f=open(name,'r')
    elif input_type is 'url':
        import requests
        f=requests.get(name).text.split('\n')
    elif input_type == "list":
        f = name
    return _gspan_to_eden(f)        
   

def _gspan_to_eden(data_str_list):
    def gspan_to_networkx(string_list):
        G=nx.Graph()
        for line in string_list:
            if line.strip():
                line_list=line.split()
                fc=line_list[0]
                #process vertices
                if fc in ['v','V'] : 
                    vid = int(line_list[1])
                    vlabel = line_list[2]
                    hvlabel = [hash(vlabel)]
                    #lowercase v indicates active viewpoint
                    if fc == 'v':
                        viewpoint = True
                    else: #uppercase v indicates no-viewpoint
                        viewpoint = False
                    G.add_node(vid, label = vlabel, hlabel = hvlabel, viewpoint = viewpoint)
                    #abstract vertices
                    if vlabel[0] == '^':
                        G.node[vid]['nesting'] = True
                    #extract the rest of the line  as a JSON string that contains all attributes
                    attribute_str=' '.join(line_list[3:])
                    if attribute_str.strip():
                        attribute_dict = json.loads(attribute_str)
                        G.node[vid].update(attribute_dict)
                #process edges
                if fc == 'e' : 
                    srcid = int(line_list[1])
                    destid = int(line_list[2])
                    elabel = line_list[3]
                    helabel=[hash(elabel)]
                    G.add_edge(srcid, destid, label = elabel, hlabel = helabel, viewpoint = True)
                    attribute_str=' '.join(line_list[4:])
                    if attribute_str.strip():
                        attribute_dict=json.loads(attribute_str)
                        G.edge[srcid][destid].update(attribute_dict)
        assert(len(G)>0),'ERROR: generated empty graph'
        return G

    string_list=[]
    for line in data_str_list:
        if line.strip():
            if line[0] in ['g','t']:
                if len(string_list) != 0:
                    yield gspan_to_networkx(string_list)
                string_list=[]
            string_list+=[line]

    if len(string_list) != 0:
        yield gspan_to_networkx(string_list)