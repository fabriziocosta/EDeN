from eden.converter.molecule import obabel
import networkx as nx
import pybel
import requests
import os.path
import time

#AID=2
AID=2401
DATA_DIR = '/home/liconj/proj/thesis/EDeN/examples/3Dmodel/data'
active_fname=DATA_DIR + '/AID%s_active.smi'%AID
inactive_fname=DATA_DIR + '/AID%s_inactive.smi'%AID

def make_iterable(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()

import datetime, time
def train_obabel_model(iterable_pos, iterable_neg, data_dir,
                       model_type = "default",
                       model_fname=None, n_iter=40, active_set_size=1000,
                       n_active_learning_iterations=3, threshold=1, train_test_split=0.7,
                       verbose=False):

    from numpy.random import randint
    from numpy.random import uniform


    global_cache = {}

    # this will be passed as an argument to the model later on
    def pre_processor(data, model_type="3d", **kwargs):

        #### Use the model_type variable from outside (?) ####
        # model_type = kwargs.get('mode', 'default')
        if model_type == "default":
            iterable = obabel.obabel_to_eden(data, **kwargs)
        elif model_type == "3d":
            iterable = obabel.obabel_to_eden3d(data, cache=global_cache, **kwargs)
        return iterable

    from eden.graph import Vectorizer
    vectorizer = Vectorizer()

    from sklearn.linear_model import SGDClassifier
    estimator = SGDClassifier(class_weight='auto', shuffle=True)

    #######3
    #create iterable from files
    ########

    from itertools import tee
    iterable_pos, iterable_pos_ = tee(iterable_pos)
    iterable_neg, iterable_neg_ = tee(iterable_neg)

    import time
    start = time.time()
    print('# positives: %d  # negatives: %d (%.1f sec %s)'%(sum(1 for x in iterable_pos_), sum(1 for x in iterable_neg_), time.time() - start, str(datetime.timedelta(seconds=(time.time() - start)))))

    iterable_pos, iterable_pos_ = tee(iterable_pos)
    iterable_neg, iterable_neg_ = tee(iterable_neg)

    #split train/test
    from eden.util import random_bipartition_iter
    iterable_pos_train, iterable_pos_test = random_bipartition_iter(iterable_pos, relative_size=train_test_split)
    iterable_neg_train, iterable_neg_test = random_bipartition_iter(iterable_neg, relative_size=train_test_split)



    #make predictive model
    from eden.model import ActiveLearningBinaryClassificationModel
    model = ActiveLearningBinaryClassificationModel(pre_processor,
                                                    estimator=estimator,
                                                    vectorizer=vectorizer,
                                                    n_jobs=2,
                                                    n_blocks = 5,
                                                    fit_vectorizer=True)

    #optimize hyperparameters and fit model

    pre_processor_parameters={'k':randint(1, 10,size=n_iter),
                             'threshold':randint(1, 10, size=n_iter),
                             'model_type':[model_type]}

    #print "pre processor parameters: " + str(pre_processor_parameters)
    vectorizer_parameters={'complexity':[2,3,4],
                           'discretization_size':randint(2, 3,size=n_iter),
                           'discretization_dimension':randint(2, 3,size=n_iter)}

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
                   n_iter=n_iter, cv=3, verbose=verbose,
                   pre_processor_parameters=pre_processor_parameters,
                   vectorizer_parameters=vectorizer_parameters,
                   estimator_parameters=estimator_parameters)

    # estimate predictive performance
    model.estimate( iterable_pos_test, iterable_neg_test )

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

pos_iterator=make_iterable(active_fname) #this is a SMILES file
neg_iterator=make_iterable(inactive_fname) #this is a SMILES file

model_fname=DATA_DIR + '/AID%s.model3d'%AID

start_time = time.time()

model = train_obabel_model(pos_iterator, neg_iterator,
                           data_dir=DATA_DIR,
                           model_type = "3d",
                           model_fname=model_fname,
                           n_iter=10,
                           active_set_size=5,
                           n_active_learning_iterations=0,
                           threshold=1,
                           train_test_split=0.8,
                           verbose=1)

end_time = time.time()
print("--- Training time: %s seconds ---" % (time.time() - start_time))
