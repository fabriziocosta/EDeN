import openbabel as ob
import pybel
import networkx as nx
import scipy.spatial.distance
import numpy as np
import subprocess
import shlex
from eden.util import read


def mol_file_to_iterable(filename=None, file_format=None):
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
    else:
        raise Exception('ERROR: unrecognized file format: %s' % file_format)


def obabel_to_eden(iterable, file_format='sdf', **options):
    """
    Takes a string list in sdf format format and yields networkx graphs.

    Parameters
    ----------
    iterable : SMILES strings containing molecular structures.

    """
    if file_format == 'sdf':
        for mol_sdf in read(iterable):
            mol = pybel.readstring("sdf", mol_sdf)
            # remove hydrogens
            mol.removeh()
            graph = obabel_to_networkx(mol)
            if len(graph):
                yield graph
    elif file_format == 'smi':
        for mol_smi in read(iterable):
            mol = pybel.readstring("smi", mol_smi)
            # remove hydrogens
            mol.removeh()
            graph = obabel_to_networkx(mol)
            if len(graph):
                graph.graph['info'] = mol_smi
                yield graph
    else:
        raise Exception('ERROR: unrecognized file format: %s' % file_format)


def obabel_to_networkx(mol):
    """
    Takes a pybel molecule object and converts it into a networkx graph.
    """
    graph = nx.Graph()
    # atoms
    for atom in mol:
        node_id = atom.idx - 1
        label = str(atom.type)
        graph.add_node(node_id, label=label)
    # bonds
    for bond in ob.OBMolBondIter(mol.OBMol):
        label = str(bond.GetBO())
        graph.add_edge(
            bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, label=label)
    return graph


##########################################################################

def obabel_to_eden_old(iterable, cache={}, format='sdf', **options):
    """
    Takes a string list in sdf format format and yields networkx graphs.

    Parameters
    ----------
    iterable : SMILES strings containing molecular structures.

    """
    if format == 'sdf':
        for mol_sdf in read(iterable):
            mol = pybel.readstring("sdf", mol_sdf)
            # remove hydrogens
            mol.removeh()
            graph = obabel_to_networkx(mol)
            if len(graph):
                yield graph
    elif format == 'smi':
        for mol_smi in read(iterable):
            # First check if the molecule has appeared before and thus is
            # already converted
            if mol_smi not in cache:
                # convert from SMILES to SDF and store in cache
                # TODO: do we assume that the input is "clean", i.e.
                # only the SMILES strings?
                command_string = 'obabel -:"' + mol_smi + '" -osdf --gen3d'
                args = shlex.split(command_string)
                sdf = subprocess.check_output(args)
                # Add the MOL object, not sdf to cache
                cache[mol_smi] = sdf
            # Convert to networkx
            graph = obabel_to_networkx(pybel.readstring('sdf', cache[mol_smi]))
            # TODO: change back to yield (below too!)
            if len(graph):
                yield graph


def obabel_to_eden3d(iterable, file_format='sdf', cache={}, split_components=True, **kwargs):
    """
    Takes an iterable file and yields the corresponding networkx graphs.

    **kwargs: arguments to be passed to other methods.
    """

    n_conf = kwargs.get('n_conf', 0)

    if file_format == 'sdf':
        if split_components:  # yield every graph separately
            for mol_sdf in read(iterable):
                mol = pybel.readstring("sdf", mol_sdf)
                mols = generate_conformers(mol.write("sdf"), n_conf)
                for molecule in mols:
                    molecule.removeh()
                    graph = obabel_to_networkx3d(molecule, **kwargs)
                    if len(graph):
                        yield graph
        else:  # construct a global graph and accumulate everything there
            global_graph = nx.Graph()
            for mol_sdf in read(iterable):
                mol = pybel.readstring("sdf", mol_sdf)
                mols = generate_conformers(mol.write("sdf"), n_conf)
                for molecule in mols:
                    molecule.removeh()
                    g = obabel_to_networkx3d(molecule, **kwargs)
                    if len(g):
                        global_graph = nx.disjoint_union(global_graph, g)
            yield global_graph

    elif file_format == 'smi':
        if split_components:  # yield every graph separately
            for mol_smi in read(iterable):
                # First check if the molecule has appeared before and thus is
                # already converted
                if mol_smi not in cache:
                    # convert from SMILES to SDF and store in cache
                    command_string = 'obabel -:"' + mol_smi + '" -osdf --gen3d'
                    args = shlex.split(command_string)
                    sdf = subprocess.check_output(args)
                    # Assume the incoming string contains only one molecule
                    # Remove warning messages generated by openbabel
                    sdf = '\n'.join(
                        [x for x in sdf.split('\n') if 'WARNING' not in x])
                    cache[mol_smi] = sdf

                mols = generate_conformers(cache[mol_smi], n_conf)
                for molecule in mols:
                    graph = obabel_to_networkx3d(molecule, **kwargs)
                    if len(graph):
                        yield graph

        else:  # construct global graph and accumulate everything there
            global_graph = nx.Graph()
            for mol_smi in read(iterable):
                # First check if the molecule has appeared before and thus is
                # already converted
                if mol_smi not in cache:
                    # convert from SMILES to SDF and store in cache
                    command_string = 'obabel -:"' + mol_smi + '" -osdf --gen3d'
                    args = shlex.split(command_string)
                    sdf = subprocess.check_output(args)
                    sdf = '\n'.join(
                        [x for x in sdf.split('\n') if 'WARNING' not in x])
                    cache[mol_smi] = sdf

                mols = generate_conformers(cache[mol_smi], n_conf)
                for molecule in mols:
                    g = obabel_to_networkx3d(molecule, **kwargs)
                    if len(g):
                        global_graph = nx.disjoint_union(global_graph, g)
            yield global_graph

    else:
        raise Exception('ERROR: unrecognized file format: %s' % file_format)


def obabel_to_networkx3d(input_mol, **kwargs):
    # similarity_fn, atom_types=None, k=3, threshold=0):
    """
    Takes a pybel molecule object and converts it into a networkx graph.

    :param input_mol: A molecule object
    :type input_mol: pybel.Molecule
    :param atom_types: A list containing the atomic number of atom types to be
     looked for in the molecule
    :type atom_types: list or None
    :param k: The number of nearest neighbors to be considered
    :type k: int
    :param label_name: the name to be used for the neighbors attribute
    :type label_name: string
    """

    graph = nx.Graph()

    # Calculate pairwise distances between all atoms:
    coords = []
    for atom in input_mol:
        coords.append(atom.coords)

    coords = np.asarray(coords)
    distances = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(coords))

    # Find the nearest neighbors for each atom
    for atom in input_mol:
        atomic_no = str(atom.type)
        node_id = atom.idx - 1
        graph.add_node(node_id)
        graph.node[node_id]['label'] = find_nearest_neighbors(
            input_mol, distances, atom.idx, **kwargs)
        graph.node[node_id]['discrete_label'] = atomic_no
        graph.node[node_id]['ID'] = node_id

    for bond in ob.OBMolBondIter(input_mol.OBMol):
        label = str(bond.GetBO())
        graph.add_edge(bond.GetBeginAtomIdx() - 1,
                       bond.GetEndAtomIdx() - 1,
                       label=label)
    # print "current graph edges: "
    # print g.edges()
    return graph


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
    atom_types = kwargs.get('atom_types', [1, 2, 8, 6, 10, 26, 7, 14, 12, 16])
    similarity_fn = kwargs.get('similarity_fn', lambda x: 1. / (x + 1))
    k = kwargs.get('k', 3)
    threshold = kwargs.get('threshold', 0)
    sorted_indices = np.argsort(distances[current_idx - 1, ])

    # Obs: nearest_atoms will contain the atom index, which starts in 0
    nearest_atoms = []

    for atomic_no in atom_types:
        # Don't remove current atom from list:
        atom_idx = [atom.idx for atom in mol if atom.atomicnum == atomic_no]
        # Indices cannot be compared directly with idx
        if len(atom_idx) >= k:
            nearest_atoms.append(
                [id for id in sorted_indices if id + 1 in atom_idx][:k])
        else:
            nearest_atoms.append(
                [id for id in sorted_indices if id + 1 in atom_idx] +
                [None] * (k - len(atom_idx)))

    # The following expression flattens the list
    nearest_atoms = [x for sublist in nearest_atoms for x in sublist]
    # Replace idx for distances, assign an arbitrarily large distance for None
    nearest_atoms = [distances[current_idx - 1, i]
                     if i is not None else 1e10 for i in nearest_atoms]
    # If a threshold value is entered, filter the list of distances
    if threshold > 0:
        nearest_atoms = [x if x <= threshold else 1e10 for x in nearest_atoms]
    # Finally apply the similarity function to the resulting list and return
    nearest_atoms = [similarity_fn(x) for x in nearest_atoms]

    return nearest_atoms


def generate_conformers(input_sdf, n_conf=10, method="rmsd"):
    """
    Given an input sdf string, call obabel to construct a specified number of
    conformers.
    """
    import subprocess
    import pybel as pb
    import re

    if n_conf == 0:
        return [pb.readstring("sdf", input_sdf)]

    command_string = 'echo "%s" | obabel -i sdf -o sdf --conformer --nconf %d\
    --score rmsd --writeconformers 2>&-' % (input_sdf, n_conf)
    sdf = subprocess.check_output(command_string, shell=True)
    # Clean the resulting output
    first_match = re.search('OpenBabel', sdf)
    clean_sdf = sdf[first_match.start():]
    # Accumulate molecules in a list
    mols = []
    # Each molecule in the sdf output begins with the 'OpenBabel' string
    matches = list(re.finditer('OpenBabel', clean_sdf))
    for i in range(len(matches) - 1):
        # The newline at the beginning is needed for obabel to recognize the
        # sdf format
        mols.append(
            pb.readstring(
                "sdf",
                '\n' + clean_sdf[matches[i].start():matches[i + 1].start()]))

    mols.append(pb.readstring("sdf", '\n' + clean_sdf[matches[-1].start():]))
    return mols


def flip_node_labels(graph, new_label_name, old_label_name):
    import networkx as nx
    # If the specified new label name doesn't exist, assume
    # that it is already the main label - do nothing
    if new_label_name not in graph.node[0].keys():
        return graph
    else:
        # Extract data from old label
        old_label_data = dict([(n, d['label'])
                               for n, d in graph.nodes_iter(data=True)])
        # Extract data from new label
        new_label_data = dict([(n, d[new_label_name])
                               for n, d in graph.nodes_iter(data=True)])
        # Swap the information
        nx.set_node_attributes(graph, 'label', new_label_data)
        nx.set_node_attributes(graph, old_label_name, old_label_data)
        # Delete information corresponding to the old label name,
        # as it will be redundant.
        for id, node in graph.nodes(data=True):
            del node[new_label_name]

        return graph
