

import datetime, time
import sys
import argparse

import requests
import os.path


def main(argv=None):
    
    ############ option processing #################
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--aid', type = int, required = True,
                        help = "Bioassay ID")
    ## Training options:
    parser.add_argument('--niter', default = 10, type = int,
                        help = "Number of training iterations")
    parser.add_argument('--asize', default = 1000, type = int,
                        help = "Active set size")
    parser.add_argument('--aiter', default = 3, type = int,
                        help = "Number of active learning iterations")
    parser.add_argument('-t', default = 10, type = int,
                        help = "Threshold value")
    parser.add_argument('--split', default = 0.7, type = float,
                        help = "Train-test split")
    parser.add_argument('-v', default = 0, type = int,
                        help = "Verbosity")
    parser.add_argument('-m', default = "default",
                        help = "Model type")
    parser.add_argument('--nconf', default = 0, type = int,
                        help = "Number of molecule conformers")
    ## Vectorizer options:
    #parser.add_argument()
        
    global args
    args = vars(parser.parse_args())
    
    AID = args['aid']

    active_fname   = 'data/AID%s_active.sdf'   % AID
    inactive_fname = 'data/AID%s_inactive.sdf' % AID

    model_fname = 'AID%s_%s.model' % (AID, args['m'])
    
    print "model name: ", model_fname
    
    #for key, value in args:
        #print key, " is of type " + str(type(value))
    
    print args
    
    model = train_obabel_model(active_fname, inactive_fname,
                               model_fname = model_fname, 
                               n_iter = args['niter'], 
                               active_set_size=500, 
                               n_active_learning_iterations=4, 
                               threshold=1, 
                               train_test_split=0.7, 
                               verbose=1)


def train_obabel_model(pos_fname, neg_fname, model_fname=None,
                       n_iter=40,
                       active_set_size=1000,
                       n_active_learning_iterations=3,
                       threshold=1,
                       train_test_split=0.7,
                       verbose=False):
    
    import networkx as nx
    import pybel
    
    
    def pre_processor( data, **args):
        return data
    
    from eden.graph import Vectorizer
    vectorizer = Vectorizer()

    from sklearn.linear_model import SGDClassifier
    # estimator = SGDClassifier(average=True, class_weight='auto', shuffle=True)
    estimator = SGDClassifier(class_weight='auto', shuffle=True)
    
    # Create iterable from files
    from eden.converter.molecule import obabel
    iterable_pos = obabel.obabel_to_eden(pos_fname)
    iterable_neg = obabel.obabel_to_eden(neg_fname)
    
    from itertools import tee
    iterable_pos, iterable_pos_ = tee(iterable_pos)
    iterable_neg, iterable_neg_ = tee(iterable_neg)
    
    import time
    start = time.time()
    print('# positives: %d  # negatives: %d (%.1f sec %s)'%(sum(1 for x in iterable_pos_), sum(1 for x in iterable_neg_), time.time() - start, str(datetime.timedelta(seconds=(time.time() - start)))))
    
    # Split train/test
    from eden.util import random_bipartition_iter
    iterable_pos_train, iterable_pos_test = random_bipartition_iter(iterable_pos, relative_size=train_test_split)
    iterable_neg_train, iterable_neg_test = random_bipartition_iter(iterable_neg, relative_size=train_test_split)



    # Make predictive model
    from eden.model import ActiveLearningBinaryClassificationModel
    model = ActiveLearningBinaryClassificationModel(
        pre_processor,
        estimator = estimator,
        vectorizer = vectorizer,
        n_jobs = 2,
        n_blocks = 2,
        fit_vectorizer = True)

    # Optimize hyperparameters and fit model
    from numpy.random import randint
    from numpy.random import uniform

    pre_processor_parameters={} 
    
    # The training time for this model is much smaller, so we can use various iterations of the
    # vectorizer
    # vectorizer_parameters={'complexity':[2,3,4,5]}
    vectorizer_parameters={'complexity':[3]}

    estimator_parameters={'n_iter':randint(5, 100, size=n_iter),
                          'penalty':['l1','l2','elasticnet'],
                          'l1_ratio':uniform(0.1,0.9, size=n_iter), 
                          'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                          'power_t':uniform(0.1, size=n_iter),
                          'alpha': [10**x for x in range(-8,-2)],
                          'eta0': [10**x for x in range(-4,-1)],
                          'learning_rate': ["invscaling", "constant", "optimal"]}

    model.optimize(iterable_pos_train, iterable_neg_train, 
                   model_name = model_fname,
                   n_active_learning_iterations = n_active_learning_iterations,
                   size_positive = -1,
                   size_negative = active_set_size,
                   n_iter=n_iter,
                   cv=3,
                   verbosity=verbose,
                   pre_processor_parameters = pre_processor_parameters, 
                   vectorizer_parameters = vectorizer_parameters, 
                   estimator_parameters = estimator_parameters)
    
    # Estimate predictive performance
    model.estimate(iterable_pos_test, iterable_neg_test)
    return model
    
    
def test_obabel_model(fname, model_fname=None):
    from eden.model import ActiveLearningBinaryClassificationModel

    model = ActiveLearningBinaryClassificationModel()
    model.load(model_fname)

    #create iterable from files
    from eden.converter.molecule import obabel
    iterable=obabel.obabel_to_eden(fname)
    
    predictions= model.decision_function( iterable )
        
    return predictions


if __name__ == "__main__":
    sys.exit(main())



