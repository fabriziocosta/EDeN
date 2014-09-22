import pybel as pb
import openbabel as ob
import networkx as nx 

def obabel_to_eden(infile, file_type='smi'):
    
    def obabel2networkx(mol):
        g = nx.Graph()
        for atom in mol:
            vlabel=atom.type
            hvlabel=[hash(vlabel)]
            g.add_node(atom.idx, label=vlabel, hlabel=hvlabel)

            edges = []
        bondorders = []
        for bond in ob.OBMolBondIter(mol.OBMol):
            elabel=bond.GetBO()
            helabel=[hash(elabel)]
            g.add_edge(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(), label=elabel, hlabel=helabel)
        return g

    
    counter=0
    string_list=[]
    for  mol in pb.readfile(file_type, infile):
        yield obabel2networkx(mol)
        counter=counter+1

