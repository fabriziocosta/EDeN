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
        for x in read(input):
            mol = pybel.readstring("sdf", x)
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
        node_id = atom.idx - 1
        label = str(atom.type)
        g.add_node(node_id, label=label)
    #bonds
        edges = []
    bondorders = []
    for bond in ob.OBMolBondIter(mol.OBMol):
        label = str(bond.GetBO())
        g.add_edge( bond.GetBeginAtomIdx()-1, bond.GetEndAtomIdx()-1, label = label )
    return g


####################################################################################


def obabel_to_eden3d(input, split_components=True, **kwargs):
    """
    Takes an input file and yields the corresponding networkx graphs.

    **kwargs: arguments to be passed to other methods.
    """
    
    n_conf = kwargs.get('n_conf', 0)
    
    if split_components: # yield every graph separately
        for x in input:
            mol = pybel.readstring("sdf", x)
            mols = generate_conformers(mol.write("sdf"), n_conf)
            for molecule in mols:
                molecule.removeh()
                G = obabel_to_networkx3d(molecule, **kwargs)    
                if len(G):
                    yield G
    else: # construct a global graph and accumulate everything there
        G_global = nx.Graph()
        for x in input:
            mol = pybel.readstring("sdf", x)
            mols = generate_conformers(mol.write("sdf"), n_conf)
            for molecule in mols:
                molecule.removeh()
                G = obabel_to_networkx3d(molecule, **kwargs)
                if len(G):
                    G_global = nx.disjoint_union(G_global, G)
        yield G_global

def obabel_smiles_to_eden3d(input, cache={}, split_components=True, **kwargs):
                     # similarity_fn=None, atom_types = [1,2,8,6,10,26,7,14,12,16], k=3, threshold=0):
    """
    Takes an input file and yields the corresponding networkx graphs.

    **kwargs: arguments to be passed to other methods.
    """
    
    # How many conformers should be looked for
    n_conf = kwargs.get('n_conf', 0)
    
    if split_components: # yield every graph separately
        for x in read(input):
            # First check if the molecule has appeared before and thus is already converted
            if x not in cache:
                # convert from SMILES to SDF and store in cache
                # TODO: do we assume that the input is "clean", i.e. only the SMILES strings?
                command_string = 'obabel -:"' + x + '" -osdf --gen3d'
                args = shlex.split(command_string)
                sdf = subprocess.check_output(args)
                # Assume the incoming string contains only one molecule
                # Remove warning messages generated by openbabel
                sdf = '\n'.join([x for x in sdf.split('\n') if 'WARNING' not in x])
                cache[x] = sdf
                # print "Molecule converted and stored"
            
            mols = generate_conformers(cache[x], n_conf)
            for molecule in mols:
                G = obabel_to_networkx3d(molecule, **kwargs)
                if len(G):
                    yield G

    else: # construct global graph and accumulate everything there
        G_global = nx.Graph()
        for x in read(input):
            # First check if the molecule has appeared before and thus is already converted
            if x not in cache:
                # convert from SMILES to SDF and store in cache
                # TODO: do we assume that the input is "clean", i.e. only the SMILES strings?
                command_string = 'obabel -:"' + x + '" -osdf --gen3d'
                args = shlex.split(command_string)
                sdf = subprocess.check_output(args)
                sdf = '\n'.join([x for x in sdf.split('\n') if 'WARNING' not in x])
                cache[x] = sdf
            
            mols = generate_conformers(cache[x], n_conf)
            for molecule in mols:
                G = obabel_to_networkx3d(molecule, **kwargs)
                if len(G):
                    G_global = nx.disjoint_union(G_global, G)
        yield G_global


####
def obabel_to_networkx3d(input_mol, **kwargs):
    # similarity_fn, atom_types=None, k=3, threshold=0):

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

    g = nx.Graph()

    # Calculate pairwise distances between all atoms:
    coords = []
    for atom in input_mol:
        coords.append(atom.coords)
    coords = np.asarray(coords)
    distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))

    # Find the nearest neighbors for each atom
    # atoms
    for atom in input_mol:
        atomic_no = str(atom.type)
        node_id = atom.idx - 1
        g.add_node(node_id)
        g.node[node_id]['label'] = find_nearest_neighbors(input_mol, distances, atom.idx, **kwargs)
        g.node[node_id]['discrete_label'] = atomic_no
        g.node[node_id]['ID'] = node_id

    for bond in ob.OBMolBondIter(input_mol.OBMol):
        label = str(bond.GetBO())
        g.add_edge(bond.GetBeginAtomIdx() - 1,
                   bond.GetEndAtomIdx() - 1,
                   label = label)
    # print "current graph edges: "
    # print g.edges()
    return g

def find_nearest_neighbors(mol, distances, current_idx, **kwargs):

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
    atom_types = kwargs.get('atom_types', [1,2,8,6,10,26,7,14,12,16])
    similarity_fn = kwargs.get('similarity_fn', lambda x: 1./(x + 1))
    k = kwargs.get('k', 3)
    threshold = kwargs.get('threshold', 0)
    # print "Value of threshold parameter: %s" % threshold
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

    #print "***** Current atom id: ", current_idx
    # The following expression flattens the list
    nearest_atoms = [x for sublist in nearest_atoms for x in sublist]
    # print "** Atom id of nearest neighbors"
    # print nearest_atoms
    # Replace idx for distances, assign an arbitrarily large distance for None
    #nearest_atoms = [distances[current_idx-1, i] if not i is None else 0 for i in nearest_atoms]
    nearest_atoms = [distances[current_idx-1, i] if not i is None else 1e10 for i in nearest_atoms]
    # print "** Distance values for nearest neighbors: "
    # print nearest_atoms
    # If a threshold value is entered, filter the list of distances
    if threshold > 0:
        nearest_atoms = [x if x <= threshold else 1e10 for x in nearest_atoms]
    # Finally apply the similarity function to the resulting list and return
    nearest_atoms = [similarity_fn(x) for x in nearest_atoms]
    # print "** Similarity values from distances: "
    # print nearest_atoms

    return nearest_atoms


def generate_conformers(input_sdf, n_conf=10, method="rmsd"):
    """
    Given an input sdf string, call obabel to construct a specified number of conformers.
    """
    import subprocess
    import pybel as pb
    import re
    
    if n_conf == 0:
        return [pb.readstring("sdf", input_sdf)]
    
    command_string = 'echo "%s" | obabel -i sdf -o sdf --conformer --nconf %d --score rmsd --writeconformers 2>&-' % (input_sdf, n_conf)
    # TODO: change this to use the piping method in subprocess (?)
    sdf = subprocess.check_output(command_string, shell=True)
    # Clean the resulting output
    first_match = re.search('OpenBabel', sdf)
    clean_sdf = sdf[first_match.start():]
    # Accumulate molecules in a list
    mols = []
    # Each molecule in the sdf output begins with the 'OpenBabel' string
    matches = list(re.finditer('OpenBabel', clean_sdf))
    for i in range(len(matches)-1):
        # The newline at the beginning is needed for obabel to recognize the sdf format
        mols.append(pb.readstring("sdf", '\n' + clean_sdf[matches[i].start():matches[i+1].start()]))
    mols.append(pb.readstring("sdf", '\n' + clean_sdf[matches[-1].start():]))
    return mols

def make_iterable(filename, file_format):
    if file_format == 'sdf':
        with open(filename) as f:
            s = ''
            for line in f:
                if line.strip() != '$$$$':
                    s = s + line
                else:
                    return_value = s + line
                    s = ''
                    yield return_value
    elif file_format == 'smi':
        with open(filename) as f:
            for line in f:
                yield line



def smiles_to_sdf(infile, outfile):
    """
    Given an input file in SMILES format, call obabel to convert to SDF format.
    """
    import subprocess
    # Should hydrogens be included?
    command_string = "obabel -ismi %s -O %s " % (infile, outfile)
    p = subprocess.call(command_string.split())

def flip_node_labels(graph, new_label_name, old_label_name):
    import networkx as nx
    # If the specified new label name doesn't exist, assume
    # that it is already the main label - do nothing
    if not new_label_name in graph.node[0].keys():
        return graph
    else:
        # Extract data from old label
        old_label_data = dict([(n, d['label']) for n, d in graph.nodes_iter(data = True)])
        # Extract data from new label
        new_label_data = dict([(n, d[new_label_name]) for n, d in graph.nodes_iter(data = True)])
        # Swap the information
        nx.set_node_attributes(graph, 'label', new_label_data)
        nx.set_node_attributes(graph, old_label_name, old_label_data)
        # Delete information corresponding to the old label name,
        # as it will be redundant.
        for id, node in graph.nodes(data=True):
            del node[new_label_name]
        
        return graph
    
    
    
    