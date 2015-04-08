import openbabel as ob
import pybel
import json
import networkx as nx
from networkx.readwrite import json_graph
import tempfile
import scipy.spatial.distance
import numpy as np

def obabel_to_eden(input, file_type = 'sdf', **options):
    """
    Takes a string list in sdf format format and yields networkx graphs.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    """
    for mol in pybel.readfile(file_type, input):
        #remove hydrogens
        mol.removeh()
        G = obabel_to_networkx(mol)
        if len(G):
            yield G

def obabel_to_networkx( mol ):
    """
    Takes a pybel molecule object and converts it into a networkx graph.
    """
    g = nx.Graph()

    #atoms
    for atom in mol:
        label = str(atom.type)
        g.add_node(atom.idx, label=label)
    #bonds
        edges = []
    bondorders = []
    for bond in ob.OBMolBondIter(mol.OBMol):
        label = str(bond.GetBO())
        g.add_edge( bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label = label )
    return g


####################################################################################


def obabel_to_eden3d(input, similarity_fn=None, atom_types = [1,2,8,6,10,26,7,14,12,16], k=3, filetype = 'sdf', split_components=True):
    """
    Takes an input file and yields the corresponding networkx graphs.
    """
    
    if similarity_fn is None:
        similarity_fn = lambda x,a: 1./(a + x)
    
    if split_components: # yield every graph separately
        for mol in pybel.readfile(filetype, input):
            # remove hydrogens - what's the reason behind this?
            # mol.removeh()
            G = obabel_to_networkx3d(mol, similarity_fn, atom_types=atom_types, k=k)
            if len(G):
                yield G
    else: # construct a global graph and accumulate everything there
        G_global = nx.Graph()
        for mol in pybel.readfile(filetype, input):
            # remove hydrogens - what's the reason behind this?
            # mol.removeh()
            G = obabel_to_networkx3d(mol, similarity_fn, atom_types=atom_types, k=k)
            if len(G):
                G_global = nx.disjoint_union(G_global, G)
        yield G_global
            
            
        


def obabel_to_networkx3d(mol, similarity_fn, atom_types=None, k=3):
    """
    Takes a pybel molecule object and converts it into a networkx graph.

    :param mol: A molecule object
    :type mol: pybel.Molecule
    :param atom_types: A list containing the atomic number of atom types to be looked for in the molecule
    :type atom_types: list or None
    :param k: The number of nearest neighbors to be considered
    :type k: int
    :param label_name: the name to be used for the neighbors attribute
    :type label_name: string
    """
    g = nx.Graph()

    if atom_types is None:
        # Most common elements in our galaxy with atomic number:
        # 1 Hydrogen
        # 2 Helium
        # 8 Oxygen
        # 6 Carbon
        # 10 Neon
        # 26 Iron
        # 7 Nitrogen
        # 14 Silicon
        # 12 Magnesium
        # 16 Sulfur
        atom_types = [1,2,8,6,10,26,7,14,12,16]

    # Calculate pairwise distances between all atoms:
    coords = []
    for atom in mol:
        coords.append(atom.coords)
    coords = np.asarray(coords)
    distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))

    # Find the nearest neighbors for each atom
    # atoms
    for atom in mol:
        label = str(atom.type)
        # print "atom index: ", atom.idx
        g.add_node(atom.idx, label=find_nearest_neighbors(mol, distances, atom.idx, k, atom_types, similarity_fn))
        g.node[atom.idx]['atom_type'] = label

    for bond in ob.OBMolBondIter(mol.OBMol):
        label = str(bond.GetBO())
        g.add_edge( bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label = label )
    # print "current graph edges: "
    # print g.edges()
    return g

def find_nearest_neighbors(mol, distances, current_idx, k, atom_types, similarity_fn):

    sorted_indices = np.argsort(distances[current_idx-1, ])

    # Obs: nearest_atoms will contain the atom index, which starts in 0
    nearest_atoms = []

    for atomic_no in atom_types:
        # Remove the current atom from this list as well
        # atom_idx = [atom.idx for atom in mol if atom.atomicnum == atomic_no and atom.idx != current_idx]
        # Don't remove it:
        atom_idx = [atom.idx for atom in mol if atom.atomicnum == atomic_no]
        # Indices cannot be compared directly with idx
        if len(atom_idx) >= k:
            nearest_atoms.append([id for id in sorted_indices if id+1 in atom_idx][:k])
        else:
            nearest_atoms.append([id for id in sorted_indices if id+1 in atom_idx] + [None]*(k-len(atom_idx)))

    # The following expression flattens the list
    nearest_atoms = [x for sublist in nearest_atoms for x in sublist]
    # Replace idx for inverse distance (similarities)
    nearest_atoms = [similarity_fn(distances[current_idx-1, i]) if not i is None else 0 for i in nearest_atoms]

    return nearest_atoms


def generate_conformers(infile, outfile, n_conformers, method):
    """
    Given an input file, call obabel to construct a specified number of conformers.
    """
    import subprocess
    command_string = "obabel %s -O %s --conformer --nconf %s --score %s --writeconformers" % \
                     (infile, outfile, n_conformers, method)
    p = subprocess.Popen(command_string.split())

def smiles_to_sdf(infile, outfile):
    """
    Given an input file in SMILES format, call obabel to convert to SDF format.
    """
    import subprocess
    # Should hydrogens be included?
    command_string = "obabel -ismi %s -O %s " % (infile, outfile)
    p = subprocess.Popen(command_string.split())
















    