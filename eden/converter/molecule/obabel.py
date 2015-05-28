import openbabel as ob
import pybel
import networkx as nx


def obabel_to_eden(input, file_type='sdf', **options):
    """
    Takes a string list in sdf format format and yields networkx graphs.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    """
    for mol in pybel.readfile(file_type, input):
        # remove hydrogens
        mol.removeh()
        G = obabel_to_networkx(mol)
        if len(G):
            yield G


def obabel_to_networkx(mol):
    """
    Takes a pybel molecule object and converts it into a networkx graph.
    """
    g = nx.Graph()
    # atoms
    for atom in mol:
        label = str(atom.type)
        g.add_node(atom.idx, label=label)
    # bonds
    for bond in ob.OBMolBondIter(mol.OBMol):
        label = str(bond.GetBO())
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label=label)
    return g
