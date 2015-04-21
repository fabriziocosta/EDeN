import openbabel as ob
import pybel
import json
import networkx as nx
from networkx.readwrite import json_graph
import tempfile
import scipy.spatial.distance
import numpy as np
import subprocess
import os
import shlex
from eden.util import read


def obabel_to_eden(input, format = 'sdf', **options):
    """
    Takes a string list in sdf format format and yields networkx graphs.

    Parameters
    ----------
    input : SMILES strings containing molecular structures.

    """
    #cache={}
    #for smi in read(input):
    #if smi in cache:
    #do openbabel with cache[smi]
    #else do 3dobabel and store mol in cache[smi]=mol
    if format == 'sdf':
        for mol in pybel.readfile("sdf", input):
            #remove hydrogens
            mol.removeh()
            G = obabel_to_networkx(mol)
            if len(G):
                yield G
    elif format == 'smi':
        for x in read(input):
            # First check if the molecule has appeared before and thus is already converted
            if x not in cache:
                # convert from SMILES to SDF and store in cache
                # TODO: do we assume that the input is "clean", i.e. only the SMILES strings?
                # command_string = 'obabel -:"' + x.split()[1] + '" -osdf --gen3d'
                # TODO: conformer generation still isn't working - is it a problem in my installation?
                # command_string = 'obabel -:"' + x + '" -osdf --gen3d --conformer --nconf 5 --score rmsd'
                command_string = 'obabel -:"' + x + '" -osdf --gen3d'
                args = shlex.split(command_string)
                sdf = subprocess.check_output(args)
                # Add the MOL object, not sdf to cache
                cache[x] = sdf
                # print "Output: "
                # print sdf
            # Convert to networkx
            G = obabel_to_networkx3d(cache[x], similarity_fn, threshold=threshold, atom_types=atom_types, k=k)
            # TODO: change back to yield (below too!)
            if len(G):
                yield G
def obabel_to_networkx(mol):
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


def obabel_to_eden3d(input, cache={}, similarity_fn=None, atom_types = [1,2,8,6,10,26,7,14,12,16], k=3, split_components=True,
                     threshold=0):
    """
    Takes an input file and yields the corresponding networkx graphs.
    """

    if similarity_fn is None:
        similarity_fn = lambda x: 1./(1e-10 + x)

    if split_components: # yield every graph separately
        for x in read(input):
            # First check if the molecule has appeared before and thus is already converted
            if x not in cache:
                # convert from SMILES to SDF and store in cache
                # TODO: do we assume that the input is "clean", i.e. only the SMILES strings?
                # command_string = 'obabel -:"' + x.split()[1] + '" -osdf --gen3d'
                # TODO: conformer generation still isn't working - is it a problem in my installation?
                # command_string = 'obabel -:"' + x + '" -osdf --gen3d --conformer --nconf 5 --score rmsd'
                command_string = 'obabel -:"' + x + '" -osdf --gen3d'
                args = shlex.split(command_string)
                sdf = subprocess.check_output(args)
                # Assume the incoming string contains only one molecule
                cache[x] = sdf
                # print "Molecule converted and stored"
            # Convert to networkx
            G = obabel_to_networkx3d(cache[x], similarity_fn, threshold=threshold, atom_types=atom_types, k=k)
            # TODO: change back to yield (below too!)
            if len(G):
                yield G
    else: # construct global graph and accumulate everything there
        G_global = nx.Graph()
        for x in read(input):
            # First check if the molecule has appeared before and thus is already converted
            if x not in cache:
                # convert from SMILES to SDF and store in cache
                # TODO: do we assume that the input is "clean", i.e. only the SMILES strings?
                # command_string = 'obabel -:"' + x.split()[1] + '" -osdf --gen3d'
                # TODO: conformer generation still isn't working - is it a problem in my installation?
                # command_string = 'obabel -:"' + x + '" -osdf --gen3d --conformer --nconf 5 --score rmsd'
                command_string = 'obabel -:"' + x + '" -osdf --gen3d'
                args = shlex.split(command_string)
                sdf = subprocess.check_output(args)
                cache[x] = sdf
            # Convert to networkx
            G = obabel_to_networkx3d(cache[x], similarity_fn, threshold=threshold, atom_types=atom_types, k=k)
            if len(G):
                G_global = nx.disjoint_union(G_global, G)
        yield G_global



def obabel_to_networkx3d(input_mol, similarity_fn, atom_types=None, k=3, threshold=0):
    """
    Takes a pybel molecule object and converts it into a networkx graph.

    :param input_mol: A molecule object
    :type input_mol: pybel.Molecule
    :param atom_types: A list containing the atomic number of atom types to be looked for in the molecule
    :type atom_types: list or None
    :param k: The number of nearest neighbors to be considered
    :type k: int
    :param label_name: the name to be used for the neighbors attribute
    :type label_name: string
    """

    input_mol = pybel.readstring(format="sdf", string=input_mol)
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
    for atom in input_mol:
        coords.append(atom.coords)
    coords = np.asarray(coords)
    distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))

    # Find the nearest neighbors for each atom
    # atoms
    for atom in input_mol:
        label = str(atom.type)
        # print "atom index: ", atom.idx
        g.add_node(atom.idx, label=find_nearest_neighbors(input_mol, distances, atom.idx, k, atom_types, similarity_fn, threshold))
        g.node[atom.idx]['atom_type'] = label

    for bond in ob.OBMolBondIter(input_mol.OBMol):
        label = str(bond.GetBO())
        g.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   label = label)
    # print "current graph edges: "
    # print g.edges()
    return g

def find_nearest_neighbors(mol, distances, current_idx, k, atom_types, similarity_fn,
                           threshold=0):

    sorted_indices = np.argsort(distances[current_idx-1, ])

    # Obs: nearest_atoms will contain the atom index, which starts in 0
    nearest_atoms = []

    for atomic_no in atom_types:
        # Don't remove current atom from list:
        atom_idx = [atom.idx for atom in mol if atom.atomicnum == atomic_no]
        # Indices cannot be compared directly with idx
        if len(atom_idx) >= k:
            nearest_atoms.append([id for id in sorted_indices if id+1 in atom_idx][:k])
        else:
            nearest_atoms.append([id for id in sorted_indices if id+1 in atom_idx] + [None]*(k-len(atom_idx)))

    # The following expression flattens the list
    nearest_atoms = [x for sublist in nearest_atoms for x in sublist]
    # Replace idx for distances
    nearest_atoms = [distances[current_idx-1, i] if not i is None else 0 for i in nearest_atoms]
    # If a threshold value is entered, filter the list of distances
    if threshold > 0:
        nearest_atoms = [x if x <= threshold else 0 for x in nearest_atoms]
    # Finally apply the similarity function to the resulting list and return
    nearest_atoms = [similarity_fn(x) for x in nearest_atoms]

    return nearest_atoms


def generate_conformers(infile, outfile, n_conformers, method):
    """
    Given an input file, call obabel to construct a specified number of conformers.
    """
    import subprocess
    command_string = "obabel %s -O %s --conformer --nconf %s --score %s --writeconformers" % \
                     (infile, outfile, n_conformers, method)
    p = subprocess.call(command_string.split())

def smiles_to_sdf(infile, outfile):
    """
    Given an input file in SMILES format, call obabel to convert to SDF format.
    """
    import subprocess
    # Should hydrogens be included?
    command_string = "obabel -ismi %s -O %s " % (infile, outfile)
    p = subprocess.call(command_string.split())

