__author__ = 'jl'


"""
binary_classification_model has the random_bipartition_iter to split the
data in train/test sets. both are then used to create the data matrices.

1. use openbabel to create multiple conformers. the converter should take as
input these parameters for openbabel (energy thresholds, number of conformers, etc)
2. implement the two possibilities:
  all graphs are constructed as disjoint union
  each one is yielded separately
3. use the random bipartition to create train/test split, then use the vectorizer to
fit (clustering) and transform (fit on train, transofmr on both test and train)

vectorizer has two params: discretization_size = number of clusters
discretization_dimension = times that the clustering is repeated before 
constructing the union over all features

compare original format and new format (measure accuracy on test sets for both)
    -> adjust parameters accordingly

thursday 26th, 15.30

bioassay = an experiment! not a compound

"""



from eden.converter.molecule import obabel
import networkx as nx
import matplotlib.pyplot as plt
import pybel


filename = "tryptophan.sdf"
# Most common elements in our galaxy with atomic number:
# 1	Hydrogen
# 2	Helium
# 8	Oxygen
# 6	Carbon
# 10 Neon
# 26 Iron
# 7	Nitrogen
# 14 Silicon
# 12 Magnesium
# 16 Sulfur

# iterable_mol = obabel.obabel_to_eden(filename)
mol = pybel.readfile("sdf", "tryptophan.sdf").next()



graph = obabel.obabel_to_networkx3d(mol)
obabel.generate_conformers('data/tryptophan.sdf', 'data/ga_tryptophan_ex.sdf', 10, 'rmsd')

