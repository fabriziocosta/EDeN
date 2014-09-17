from sklearn.kernel_approximation import Nystroem
import json
import networkx as nx

from eden.hasher import WTA_hash



def gspan_to_eden(name, is_file=False, is_url=False, hasher=WTA_hash()):
    if is_file :
        with open(name,'r') as f:
            return _gspan_to_eden(f,hasher)
    elif is_url :
        import requests
        f=requests.get(name).text.split('\n')
        return _gspan_to_eden(f,hasher)
    else :
        return _gspan_to_eden(name,hasher)



def _gspan_to_eden(data_str_list, hasher=WTA_hash()):
    def gspan_to_networkx(string_list):
        G=nx.Graph()
        for line in string_list:
            if len(line) > 0 :
                line_list=line.split()
                fc=line_list[0]
                if fc == 'v' : #insert vertex
                    vid = int(line_list[1])
                    vlabel = line_list[2]
                    hvlabel = [hash(vlabel)]
                    G.add_node(vid, label=vlabel, hlabel=hvlabel)
                    #extract the rest of the line  as a JSON string
                    attribute_str=' '.join(line_list[3:])
                    if len(attribute_str) > 0:
                        attribute_dict = json.loads(attribute_str)
                        G.node[vid].update(attribute_dict)
                if fc == 'e' : #insert edge
                    srcid = int(line_list[1])
                    destid = int(line_list[2])
                    elabel = line_list[3]
                    helabel=[hash(elabel)]
                    G.add_edge(srcid,destid, label=elabel, hlabel=helabel)
                    attribute_str=' '.join(line_list[4:])
                    if len(attribute_str) > 0:
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