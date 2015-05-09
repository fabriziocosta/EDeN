

import datetime, time
import sys
import argparse

import requests
import os.path

import logging
logger = logging.getLogger('root.%s' % (__name__))
hdl = logging.FileHandler('log/train.log')
logger.addHandler(hdl)
logger.setLevel(logging.INFO)


def main(argv=None):
    
    ############ option processing #################
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--aid', type = int, required = True,
                        help = "Bioassay ID")
    ## Training options:
    parser.add_argument('--niter', default = 10, type = int,
                        help = "Number of training iterations")
    parser.add_argument('--asize', default = 500, type = int,
                        help = "Active set size")
    parser.add_argument('--aiter', default = 4, type = int,
                        help = "Number of active learning iterations")
    parser.add_argument('-t', default = 1, type = int,
                        help = "Threshold value")
    parser.add_argument('--split', default = 0.8, type = float,
                        help = "Train-test split")
    parser.add_argument('-v', default = 1, type = int,
                        help = "Verbosity")
    parser.add_argument('-m', default = "default",
                        help = "Model type")
    ## Other options:
    parser.add_argument('--nconf', default = 0, type = int,
                        help = "Number of molecule conformers")
    #parser.add_argument()
        
    global args
    args = vars(parser.parse_args())
    
    AID = args['aid']
    mode = args['m']
    
    
    active_fname   = 'data/AID%s_active.sdf'   % AID
    inactive_fname = 'data/AID%s_inactive.sdf' % AID

    model_fname = 'models/AID%s_%s.model' % (AID, mode)
    
    
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + 
                " --- Starting model training for AID %s, %s mode" % (AID, mode))
    
    
    model = train_obabel_model(active_fname, inactive_fname, mode,
                               model_fname = model_fname, 
                               n_iter = args['niter'], 
                               active_set_size= args['asize'], 
                               n_active_learning_iterations = args['aiter'], 
                               threshold = args['t'], 
                               train_test_split = args['split'], 
                               verbose = args['v'])
    
    
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + 
                " --- Model training finished.")


def train_obabel_model(pos_fname, neg_fname, model_type,
                       model_fname=None,
                       n_iter=40,
                       active_set_size=1000,
                       n_active_learning_iterations=3,
                       threshold=1,
                       train_test_split=0.7,
                       verbose=False):
    
    import networkx as nx
    import pybel
    
    if model_type == "default":
        def pre_processor( data, **args):
            return data
        
        
        
        
    elif model_type == "3d":
        def pre_processor(data, model_type="3d", **kwargs):
            iterable = obabel.obabel_to_eden3d(data, **kwargs)
            return iterable
    
    
    ### the definition below does not work. why, I have no idea - it leads to 
    ### python trying to pickle an unpickleable fortran ddot object...
    
    #def pre_processor(data, model_type = model_type, **kwargs):
        ## model_type = kwargs.get('mode', 'default')
        #print "preprocessor model type : ", model_type
        #if model_type == "default":
            #return data
        #elif model_type == "3d":
            #iterable = obabel.obabel_to_eden3d(data, **kwargs)
            #return iterable
        
    
    
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
        n_jobs = 1,
        n_blocks = 1,
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



