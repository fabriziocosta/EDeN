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
import pybel
import requests
import os.path

# Define functions to obtain data from server
def get_compounds(fname, size, listkey):
    PROLOG='https://pubchem.ncbi.nlm.nih.gov/rest/pug/'
    with open(fname,'w') as file_handle:
        stepsize=50
        index_start=0
        for chunk, index_end in enumerate(range(0,size+stepsize,stepsize)):
            if index_end is not 0 :
                print 'Chunk %s) Processing compounds %s to %s (of a total of %s)' % (chunk, index_start, index_end-1, size)
                RESTQ = PROLOG + 'compound/listkey/' + str(listkey) + '/SDF?&listkey_start=' + str(index_start) + '&listkey_count=' + str(stepsize)
                reply=requests.get(RESTQ)
                file_handle.write(reply.text)
            index_start = index_end
        print 'compounds available in file: ', fname


def get_assay(assay_id):
    PROLOG='https://pubchem.ncbi.nlm.nih.gov/rest/pug/'
    AID=str(assay_id)
    #active
    RESTQ = PROLOG + 'assay/aid/' + AID + '/cids/JSON?cids_type=active&list_return=listkey'
    reply=requests.get(RESTQ)
    #extract the listkey
    active_listkey = reply.json()['IdentifierList']['ListKey']
    active_size = reply.json()['IdentifierList']['Size']
    active_fname = 'data/AID'+AID+'_active.sdf'
    get_compounds(fname=active_fname, size=active_size, listkey=active_listkey)

    #inactive
    RESTQ = PROLOG + 'assay/aid/' + AID + '/cids/JSON?cids_type=inactive&list_return=listkey'
    reply=requests.get(RESTQ)
    #extract the listkey
    inactive_listkey = reply.json()['IdentifierList']['ListKey']
    inactive_size = reply.json()['IdentifierList']['Size']
    inactive_fname = 'data/AID'+AID+'_inactive.sdf'
    get_compounds(fname=inactive_fname, size=inactive_size, listkey=inactive_listkey)

    return (active_fname,inactive_fname)

# Get the data from server

AID=825
READ_FROM_FILE=True
DATA_DIR = 'data'
if READ_FROM_FILE:
    active_fname=DATA_DIR + '/AID%s_active.sdf'%AID
    inactive_fname=DATA_DIR + '/AID%s_inactive.sdf'%AID
else:
    active_fname, inactive_fname = get_assay(AID)

# Generate conformers from data:
# Active compounds
if not os.path.exists(DATA_DIR + '/conf_AID%s_active.sdf'%AID):
    obabel.generate_conformers(active_fname, active_conf, 10, 'rmsd')
# Inactve compounds
if not os.path.exists(DATA_DIR + '/conf_AID%s_inactive.sdf'%AID):
    obabel.generate_conformers(active_fname, inactive_conf, 10, 'rmsd')




# Functions for training and testing the model:
import datetime, time
def train_obabel_model(pos_fname, neg_fname, model_type = "default",
                       model_fname=None, n_iter=40, active_set_size=1000,
                       n_active_learning_iterations=3, threshold=1, train_test_split=0.7,
                       verbose=False):
    # this will be passed as an argument to the model later on
    def pre_processor( data, **args):
        return data

    from eden.graph import Vectorizer
    vectorizer = Vectorizer()

    from sklearn.linear_model import SGDClassifier
    estimator = SGDClassifier(class_weight='auto', shuffle=True)

    #create iterable from files
    from eden.converter.molecule import obabel
    if model_type == "default":
        iterable_pos=obabel.obabel_to_eden(pos_fname)
        iterable_neg=obabel.obabel_to_eden(neg_fname)
    elif model_type == "3d":
        iterable_pos=obabel.obabel_to_eden3d(pos_fname)
        iterable_neg=obabel.obabel_to_eden3d(neg_fname)


    from itertools import tee
    iterable_pos, iterable_pos_ = tee(iterable_pos)
    iterable_neg, iterable_neg_ = tee(iterable_neg)

    import time
    start = time.time()
    print('# positives: %d  # negatives: %d (%.1f sec %s)'%(sum(1 for x in iterable_pos_), sum(1 for x in iterable_neg_), time.time() - start, str(datetime.timedelta(seconds=(time.time() - start)))))

    #split train/test
    from eden.util import random_bipartition_iter
    iterable_pos_train, iterable_pos_test = random_bipartition_iter(iterable_pos, relative_size=train_test_split)
    iterable_neg_train, iterable_neg_test = random_bipartition_iter(iterable_neg, relative_size=train_test_split)



    #make predictive model
    from eden.model import ActiveLearningBinaryClassificationModel
    model = ActiveLearningBinaryClassificationModel( pre_processor, estimator=estimator, vectorizer=vectorizer )

    #optimize hyperparameters and fit model
    from numpy.random import randint
    from numpy.random import uniform

    pre_processor_parameters={}

    vectorizer_parameters={'complexity':[4]}

    estimator_parameters={'n_iter':randint(5, 100, size=n_iter),
                          'penalty':['l1','l2','elasticnet'],
                          'l1_ratio':uniform(0.1,0.9, size=n_iter),
                          'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                          'power_t':uniform(0.1, size=n_iter),
                          'alpha': [10**x for x in range(-8,-2)],
                          'eta0': [10**x for x in range(-4,-1)],
                          'learning_rate': ["invscaling", "constant", "optimal"]}

    model.optimize(iterable_pos_train, iterable_neg_train,
                   model_name=model_fname,
                   n_active_learning_iterations=n_active_learning_iterations,
                   size_positive=-1,
                   size_negative=active_set_size,
                   n_iter=n_iter, cv=3, n_jobs=1, verbose=verbose,
                   pre_processor_parameters=pre_processor_parameters,
                   vectorizer_parameters=vectorizer_parameters,
                   estimator_parameters=estimator_parameters)

    #estimate predictive performance
    model.estimate( iterable_pos_test, iterable_neg_test, cv=5 )
    return model

def test_obabel_model(fname, model_type = "default", model_fname=None):
    from eden.model import ActiveLearningBinaryClassificationModel

    model = ActiveLearningBinaryClassificationModel()
    model.load(model_fname)

    #create iterable from files
    from eden.converter.molecule import obabel
    if model_type == "default":
        iterable=obabel.obabel_to_eden(fname)
    elif model_type == "3d":
        iterable=obabel.obabel_to_eden3d(fname)

    predictions= model.decision_function( iterable )

    return predictions

# Train the 3d model

model_fname='AID%s.model'%AID
model = train_obabel_model(active_fname, inactive_fname,
                           model_type = "3d",
                           model_fname=model_fname,
                           n_iter=40,
                           active_set_size=500,
                           n_active_learning_iterations=4,
                           threshold=1,
                           train_test_split=0.8,
                           verbose=0)


