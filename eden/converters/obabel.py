import openbabel as ob
import pybel as pb
import json
import networkx as nx
from networkx.readwrite import json_graph

def obabel_to_eden(name, input_type='file'):
    """
    Takes a string list in sdf format format and yields networkx graphs.

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
         f = name
    elif input_type is 'url':
        import requests
        rf=requests.get(name).text.split('\n')
        tf = NamedTemporaryFile(delete=False)
        for line in rf:
            tf.write(line)
        tf.close()
        f = tf.name
    elif input_type == "list":
        tf = NamedTemporaryFile(delete=False)
        for line in name:
            tf.write(line)
        tf.close()
        f = tf.name
    return _obabel_to_eden(f) 


def _obabel_to_eden(infile, file_type='sdf'):
    
    def obabel2networkx(mol):
        g = nx.Graph()
        #atoms
        for atom in mol:
            vlabel = atom.type
            hvlabel=[hash(str(atom.atomicnum))]
            g.add_node(atom.idx, label=vlabel, hlabel=hvlabel, viewpoint = True)
        #bonds
            edges = []
        bondorders = []
        for bond in ob.OBMolBondIter(mol.OBMol):
            elabel=bond.GetBO()
            helabel=[hash(str(elabel))]
            g.add_edge(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(), label=elabel, hlabel=helabel, viewpoint = True)
        return g

    
    for mol in pb.readfile(file_type, infile):
        #remove hydrogens
        mol.removeh()
        yield obabel2networkx(mol)