import openbabel as ob
import pybel
import json
import networkx as nx
from networkx.readwrite import json_graph

def obabel_to_eden(input, file_type = 'sdf', **options):
    """
    Takes a string list in sdf format format and yields networkx graphs.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    """

    if type( input ) == file:
        input.close()
        input_path = input.name
    else:
        temp_file = tempfile.NamedTemporaryFile( delete = False )
        for line in temp_file:
            temp_file.write(line)
        temp_file.close()
        input_path = temp_file.name

    for mol in pybel.readfile(file_type, input_path):
        #remove hydrogens
        mol.removeh()
        yield obabel_to_networkx(mol)


def obabel_to_networkx( mol ):
    """
    Takes a pybel molecule object and converts it into a networkx graph.
    """
    g = nx.Graph()
    #atoms
    for atom in mol:
        vlabel = atom.type
        hvlabel=[hash(str(atom.atomicnum))]
        g.add_node(atom.idx, label=vlabel, hlabel=hvlabel)
    #bonds
        edges = []
    bondorders = []
    for bond in ob.OBMolBondIter(mol.OBMol):
        elabel = bond.GetBO()
        helabel = [hash(str(elabel))]
        g.add_edge( bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label = elabel, hlabel = helabel )
    return g
