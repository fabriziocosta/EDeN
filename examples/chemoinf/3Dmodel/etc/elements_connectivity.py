# -*- coding: utf-8 -*-
import requests
import os
import pickle

os.chdir('/Users/jl/uni-freiburg/thesis/EDeN/etc/')

# Open file connections
if not os.path.exists('structs.sdf'):
    with open('cids.txt', 'r') as f, open('structs.sdf', 'w') as outfile:
        PROLOG = "http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        clean_ids = []
        for id in f:
            # Remove whiespace and quoations
            clean_ids.append(id.strip().strip('"'))
            RESTQ = PROLOG + str(clean_ids[-1]) + "/record/SDF/?record_type=2d&response_type=save"
            reply = requests.get(RESTQ)
            outfile.write(reply.text)


########################
# Read the data using pybel and extract the necessary infomation
import pybel as pb
import openbabel as ob
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

atom_types = [1,2,8,6,10,26,7,14,12,16]

atoms = []
b = {key: [] for key in atom_types}

for molecule in pb.readfile("sdf", "structs.sdf"):
    for atom in molecule:
#        print atom.atomicnum, " ---- valence of: ", atom.valence 
        if atom.atomicnum in atom_types:
            b[atom.atomicnum].append(atom.valence)

# Save this dictionary for further use:
pickle.dump(b, open("bonds.p", "wb"))

b_counts = {key: Counter(b[key]) for key in atom_types}

plt.bar(left=b_counts[8].keys(), height=b_counts[8].values())

for i in atom_types:
    plt.bar(left=b_counts[i].keys(), height=b_counts[i].values())

#bonds = pd.DataFrame(index = clean_ids, columns = atom_types)
