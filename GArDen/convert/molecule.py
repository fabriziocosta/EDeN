'''
interface between networkx graphs and the world of chemistry via rdkit

functionality:
# gernerating networkx graphs:
sdf_to_nx(file.sdf)
smi_to_nx(file.smi)
smiles_to_nx(smilesstringlist)

# graph out:
draw(nx)
nx_to_smi(graphlist, path_to_file.smi)


#bonus: garden style transformer.
class MoleculeToGraph
'''
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from IPython.core.display import display
import networkx as nx
import eden.graph as edengraphtools

import logging
logger = logging.getLogger(__name__)


class MoleculeToGraph(BaseEstimator, TransformerMixin):
    def __init__(self, file_format='sdf'):
        """Constructor.

        valid 'file_format' strings and what the transformer will expect
        smiles: list of smiles strings
        smi: path to .smi file
        sdf: pat to .sdf file
        """
        self.file_format = file_format

    def transform(self, data):
        """Transform."""
        try:
            if self.file_format == 'smiles':
                graphs = smiles_to_nx(data)
            elif self.file_format == 'smi':
                graphs = smi_to_nx(data)
            elif self.file_format == 'sdf':
                graphs = sdf_to_nx(data)
            else:
                raise Exception('file_format must be smiles smi or sdf')
            for graph in graphs:
                yield graph

        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


def set_coordinates(chemlist):
    for m in chemlist:
        if m:
            # updateprops fixes "RuntimeError: Pre-condition Violation"
            m.UpdatePropertyCache(strict=False)
            tmp = AllChem.Compute2DCoords(m)
        else:
            print '''set coordinates failed..'''


def draw(graphs, n_graphs_per_line=5, size=250, title_key=None, titles=None, smiles=False):
    # we want a list of graphs
    if isinstance(graphs, nx.Graph):
        print "give me a list of graphs"

    # make molecule objects
    chem = [nx_to_rdkit(graph) for graph in graphs]
    # print chem
    # calculate coordinates:
    set_coordinates(chem)

    # take care of the subtitle of each graph
    if title_key:
        legend = [g.graph[title_key] for g in graphs]
    elif titles:
        legend = titles
    else:
        legend = [str(i) for i in range(len(graphs))]

    if smiles == False:
        # make the image
        image = Draw.MolsToGridImage(chem, molsPerRow=n_graphs_per_line, subImgSize=(size, size), legends=legend)
        # display on the spot
        display(image)
    else:
        for m in chem:
            print Chem.MolToSmiles(m)
        print '\n'


def nx_to_smi(graphs, file):
    chem = [nx_to_rdkit(graph) for graph in graphs]
    smis = [Chem.MolToSmiles(m) for m in chem]
    with open(file, 'w') as f:
        f.write('\n'.join(smis))


def sdf_to_nx(file):
    # read sdf file
    suppl = Chem.SDMolSupplier(file)
    for mol in suppl:
        yield rdkmol_to_nx(mol)


def smi_to_nx(file):
    # read smi file
    suppl = Chem.SmilesMolSupplier(file)
    for mol in suppl:
        yield rdkmol_to_nx(mol)


def rdkmol_to_nx(mol):
    #  rdkit-mol object to nx.graph
    graph = nx.Graph()
    for e in mol.GetAtoms():
        graph.add_node(e.GetIdx(), label=e.GetSymbol())
    for b in mol.GetBonds():
        graph.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), label=str(int(b.GetBondTypeAsDouble())))
    return graph


def smiles_to_nx(smileslist):
    for smile in smileslist:
        mol = Chem.MolFromSmiles(smile)
        yield rdkmol_to_nx(mol)


def nx_to_rdkit(nx):
    molstring = graph_to_molfile(nx)
    return Chem.MolFromMolBlock(molstring, sanitize=False)


def graph_to_molfile(graph):
    '''
    helper of nx_to_chem

    Parameters
    ----------
    graph: nx.graph

    Returns
    -------
        sdf_string

    taken from eden. differences to eden:
    atom_line += d['label'].ljust(3) <- this is changed from eden.
    sdf_string += 'M  END' <- needs 2 spaces!
    '''
    symbols = {'1': 'H',
               '2': 'He',
               '3': 'Li',
               '4': 'Be',
               '5': 'B',
               '6': 'C',
               '7': 'N',
               '8': 'O',
               '9': 'F',
               '10': 'Ne',
               '11': 'Na',
               '12': 'Mg',
               '13': 'Al',
               '14': 'Si',
               '15': 'P',
               '16': 'S',
               '17': 'Cl',
               '18': 'Ar',
               '19': 'K',
               '20': 'Ca',
               '21': 'Sc',
               '22': 'Ti',
               '23': 'V',
               '24': 'Cr',
               '25': 'Mn',
               '26': 'Fe',
               '27': 'Co',
               '28': 'Ni',
               '29': 'Cu',
               '30': 'Zn',
               '31': 'Ga',
               '32': 'Ge',
               '33': 'As',
               '34': 'Se',
               '35': 'Br',
               '36': 'Kr',
               '37': 'Rb',
               '38': 'Sr',
               '39': 'Y',
               '40': 'Zr',
               '41': 'Nb',
               '42': 'Mo',
               '43': 'Tc',
               '44': 'Ru',
               '45': 'Rh',
               '46': 'Pd',
               '47': 'Ag',
               '48': 'Cd',
               '49': 'In',
               '50': 'Sn',
               '51': 'Sb',
               '52': 'Te',
               '53': 'I',
               '54': 'Xe',
               '55': 'Cs',
               '56': 'Ba',
               '57': 'La',
               '58': 'Ce',
               '59': 'Pr',
               '60': 'Nd',
               '61': 'Pm',
               '62': 'Sm',
               '63': 'Eu',
               '64': 'Gd',
               '65': 'Tb',
               '66': 'Dy',
               '67': 'Ho',
               '68': 'Er',
               '69': 'Tm',
               '70': 'Yb',
               '71': 'Lu',
               '72': 'Hf',
               '73': 'Ta',
               '74': 'W',
               '75': 'Re',
               '76': 'Os',
               '77': 'Ir',
               '78': 'Pt',
               '79': 'Au',
               '80': 'Hg',
               '81': 'Tl',
               '82': 'Pb',
               '83': 'Bi',
               '84': 'Po',
               '85': 'At',
               '86': 'Rn',
               '87': 'Fr',
               '88': 'Ra',
               '89': 'Ac',
               '90': 'Th',
               '91': 'Pa',
               '92': 'U',
               '93': 'Np',
               '94': 'Pu',
               '95': 'Am',
               '96': 'Cm',
               '97': 'Bk',
               '98': 'Cf',
               '99': 'Es',
               '100': 'Fm',
               '101': 'Md',
               '102': 'No',
               '103': 'Lr',
               '104': 'Rf',
               '105': 'Db',
               '106': 'Sg',
               '107': 'Bh',
               '108': 'Hs',
               '109': 'Mt',
               '110': 'Ds',
               '111': 'Rg',
               '112': 'Uub',
               '113': 'Uut',
               '114': 'Uuq',
               '115': 'Uup',
               '116': 'Uuh',
               '117': 'Uus',
               '118': 'Uuo'}

    graph = edengraphtools._revert_edge_to_vertex_transform(graph)
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default', label_attribute=None)
    # creating an SDF file from graph:
    # The header block, i.e. the first three lines, may be empty:
    sdf_string = "Networkx graph to molfile\n\n\n"

    # After the header block comes the connection table:
    # First the counts line - step by step
    counts_line = ""
    # Number of atoms

    counts_line += str(len(graph.nodes())).rjust(3)
    # Number of bonds
    counts_line += str(len(graph.edges())).rjust(3)
    # Number of atom lists
    counts_line += '  0'
    # Three blank spaces, then the chirality flag
    counts_line += '     1'
    # Five identical blocks
    counts_line += '  0' * 5
    # Finish with 0999 V2000
    counts_line += '999 V2000\n'
    sdf_string += counts_line

    # Atom block - this contains one atom line per atom in the molecule
    for n, d in graph.nodes_iter(data=True):
        atom_line = ''
        # Set all coordinates to 0
        atom_line += '    0.0000    0.0000    0.0000 '
        # Atom symbol: it should be the entry from the periodic table, using
        # atom type for now
        # atom_line += symbols.get(d['discrete_label']).ljust(3)
        atom_line += d['label'].ljust(3)
        # Lots of unnecessary atomic information:
        atom_line += ' 0  0  0  0  0  0  0  0  0  0  0  0\n'
        sdf_string += atom_line

    # Bond block
    for i, j, k in graph.edges_iter(data=True):
        edge_line = ''
        # Use the stored atom ids for the bonds, plus one
        edge_line += str(i + 1).rjust(3) + \
                     str(j + 1).rjust(3) + k['label'].rjust(3)
        # More information
        edge_line += '  0  0  0  0\n'
        sdf_string += edge_line

    sdf_string += 'M  END'
    # sdf_string += 'M END\n\n$$$$'
    return sdf_string
