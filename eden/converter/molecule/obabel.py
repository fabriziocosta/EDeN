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



def generate_conformers(infile, outfile, n_conformers, method):
    """
    Given an input file, call obabel to construct a specified number of conformers.
    """
    command_string = "obabel %s -O %s --conformer --nconf %s --score %s --writeconformers" % \
                     (infile, outfile, n_conformers, method)
    command_string = command_string.split()
    import subprocess
    print command_string
    p = subprocess.Popen(command_string)



def obabel_to_eden3d(input, filetype = 'sdf'):
    """
    Takes an input file and yields the corresponding networkx graphs.
    """
    for mol in pybel.readfile(filetype, input):

        # remove hydrogens - what's the reason behind this?
        # mol.removeh()
        G = obabel_to_networkx3d(mol)
        if len(G):
            yield G



def obabel_to_networkx3d(mol, atom_types=None, k=3):
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
    # print "coordinates: "
    # print coords.shape
    distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))

    # Find the nearest neighbors for each atom
    def find_nearest_neighbors(current_idx, k, atom_types):

        # Remove the first element, it will always be the current atom:
        sorted_indices = np.argsort(distances[current_idx-1, ])[1:]
        # print "sorted indices:"
        # print sorted_indices
        # Obs: nearest_atoms will contain the atom index, which starts in 0
        nearest_atoms = []

        for atomic_no in atom_types:
            # Remove the current atom from this list as well
            atom_idx = [atom.idx for atom in mol if atom.atomicnum == atomic_no and atom.idx != current_idx]
            # print "current atom type = ", atomic_no
            # print "atom idx to be selected from: ", atom_idx
            # Indices cannot be compared directly with idx
            if len(atom_idx) >= k:
                nearest_atoms.append([id for id in sorted_indices if id+1 in atom_idx][:k])
            else:
                nearest_atoms.append([id for id in sorted_indices if id+1 in atom_idx] + [None]*(k-len(atom_idx)))

        # print "original list with nearest atom indices:"
        # print nearest_atoms
        
        # The following expression flattens the list
        nearest_atoms = [x for sublist in nearest_atoms for x in sublist]
        # Replace idx for inverse distance (similarities)
        # print "distances: "
        # print distances[current_idx-1, ]
        # print "indices: "
        # print [[current_idx-1, i] if not i is None else 0 for i in nearest_atoms]

        # print "selected values: "
        # print [distances[current_idx-1, i] for i in nearest_atoms if not i is None]
        # print [distances[current_idx-1, i] if not i is None else 0 for i in nearest_atoms]

        nearest_atoms = [1./distances[current_idx-1, i] if not i is None else 0 for i in nearest_atoms]

        return nearest_atoms

    #atoms
    for atom in mol:
        label = str(atom.type)
        # print "atom index: ", atom.idx
        g.add_node(atom.idx, label=find_nearest_neighbors(atom.idx, k, atom_types))
        g.node[atom.idx]['atom_type'] = str(atom.type)

    for bond in ob.OBMolBondIter(mol.OBMol):
        label = str(bond.GetBO())
        g.add_edge( bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label = label )
    # print "current graph edges: "
    # print g.edges()
    return g