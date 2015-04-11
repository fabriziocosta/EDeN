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

class OBabelConverter(object):
    """A class that holds all the methods for conversion between openbabel
    data and networkx/eden data. A caching system is implemented for conversion
    from SMILES data to three-dimensional structures.
    
    """
    
    def __init__(self):
        # super(OBabelConverter, self).__init__()
        self.cache = {}
        
        
    def obabel_to_eden(self, input, file_type = 'sdf', **options):
        """
        Takes a string list in sdf format format and yields networkx graphs.

        Parameters
        ----------
        input : string
            A pointer to the data source.

        """
        #cache={}
        #for smi in read(input):
        #if smi in cache:
        #do openbabel with cache[smi]
        #else do 3dobabel and store mol in cache[smi]=mol
        for mol in pybel.readfile(file_type, input):
            #remove hydrogens
            mol.removeh()
            G = self.obabel_to_networkx(mol)
            if len(G):
                yield G

    def obabel_to_networkx(self, mol):
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


    def obabel_to_eden3d(self, input, similarity_fn=None, atom_types = [1,2,8,6,10,26,7,14,12,16], k=3, split_components=True):
        """
        Takes an input file and yields the corresponding networkx graphs.
        """

        if similarity_fn is None:
            similarity_fn = lambda x: 1./(1e-10 + x)

        if split_components: # yield every graph separately
            for x in read(input):
                # First check if the molecule has appeared before and thus is already converted
                if x not in self.cache:
                    # convert from SMILES to SDF and store in cache
                    # TODO: do we assume that the input is "clean", i.e. only the SMILES strings?
                    # command_string = 'obabel -:"' + x.split()[1] + '" -osdf --gen3d'
                    # TODO: conformer generation still isn't working - is it a problem in my installation?
                    # command_string = 'obabel -:"' + x + '" -osdf --gen3d --conformer --nconf 5 --score rmsd'
                    command_string = 'obabel -:"' + x + '" -osdf --gen3d'
                    args = shlex.split(command_string)
                    sdf = subprocess.check_output(args)
                    self.cache[x] = sdf
                    # print "Output: "
                    # print sdf
                # Convert to networkx
                G = self.obabel_to_networkx3d(self.cache[x], similarity_fn, atom_types=atom_types, k=k)
                # TODO: change back to yield (below too!)
                if len(G):
                    return G
        else: # construct global graph and accumulate everything there
            G_global = nx.Graph()
            for x in read(input):
                # First check if the molecule has appeared before and thus is already converted
                if x not in self.cache:
                    # convert from SMILES to SDF and store in cache
                    # TODO: do we assume that the input is "clean", i.e. only the SMILES strings?
                    # command_string = 'obabel -:"' + x.split()[1] + '" -osdf --gen3d'
                    # TODO: conformer generation still isn't working - is it a problem in my installation?
                    # command_string = 'obabel -:"' + x + '" -osdf --gen3d --conformer --nconf 5 --score rmsd'
                    command_string = 'obabel -:"' + x + '" -osdf --gen3d'
                    args = shlex.split(command_string)
                    sdf = subprocess.check_output(args)
                    self.cache[x] = sdf
                # Convert to networkx
                G = self.obabel_to_networkx3d(self.cache[x], similarity_fn, atom_types=atom_types, k=k)
                if len(G):
                    G_global = nx.disjoint_union(G_global, G)
            return G_global



    def obabel_to_networkx3d(self, mol, similarity_fn, atom_types=None, k=3):
        """
        Takes a pybel molecule object and converts it into a networkx graph.

        :param mol: A string containing SDF data
        :type mol: string
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

        # Assume the incoming string contains only one molecule
        mol = pybel.readstring(format="sdf", string=mol)

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
            g.add_node(atom.idx, label=self.find_nearest_neighbors(mol, distances, atom.idx, k, atom_types, similarity_fn))
            g.node[atom.idx]['atom_type'] = label

        for bond in ob.OBMolBondIter(mol.OBMol):
            label = str(bond.GetBO())
            g.add_edge( bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label = label )
        # print "current graph edges: "
        # print g.edges()
        return g

    def find_nearest_neighbors(self, mol, distances, current_idx, k, atom_types, similarity_fn):

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


    def generate_conformers(self, infile, outfile, n_conformers, method):
        """
        Given an input file, call obabel to construct a specified number of conformers.
        """
        import subprocess
        command_string = "obabel %s -O %s --conformer --nconf %s --score %s --writeconformers" % \
                         (infile, outfile, n_conformers, method)
        p = subprocess.call(command_string.split())

    def smiles_to_sdf(self, infile, outfile):
        """
        Given an input file in SMILES format, call obabel to convert to SDF format.
        """
        import subprocess
        # Should hydrogens be included?
        command_string = "obabel -ismi %s -O %s " % (infile, outfile)
        p = subprocess.call(command_string.split())
















    