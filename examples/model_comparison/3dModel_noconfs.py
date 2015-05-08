
# coding: utf-8

# In[1]:

from eden.converter.molecule import obabel
import networkx as nx
import pybel
import requests
import os.path

import logging
logger = logging.getLogger('root.%s' % (__name__))
hdl = logging.FileHandler('log/train.log')
logger.addHandler(hdl)
logger.setLevel(logging.INFO)


#AID=1
AID=1905
#DATA_DIR = '/home/liconj/proj/thesis/EDeN/examples/3Dmodel/data'
active_fname = 'data/AID%s_active.sdf'%AID
inactive_fname = 'data/AID%s_inactive.sdf'%AID

# In[3]:

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


# In[20]:
import datetime, time
def train_obabel_model(iterable_pos, iterable_neg, pre_processor_parameters,
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
            iterable = obabel.obabel_to_eden3d(data, **kwargs)
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
                                                    n_blocks = 10,
                                                    fit_vectorizer=True)

    #optimize hyperparameters and fit model

    #print "pre processor parameters: " + str(pre_processor_parameters)
    vectorizer_parameters={'complexity':[4]}

    estimator_parameters={'n_iter':randint(5, 100, size=n_iter),
                          'penalty':['l1','l2','elasticnet'],
                          'l1_ratio':uniform(0.1,0.9, size=n_iter),
                          'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                          'power_t':uniform(0.1, size=n_iter),
                          'alpha': [10**x for x in range(-8,-2)],
                          'eta0': [10**x for x in range(-4,-1)],
                          'learning_rate': ["invscaling", "constant", "optimal"]}

    print "calling optimizer.."
    model.optimize(iterable_pos_train, iterable_neg_train,
                   model_name=model_fname,
                   n_active_learning_iterations=n_active_learning_iterations,
                   size_positive=-1,
                   size_negative=active_set_size,
                   n_iter=n_iter, cv=3,
                   pre_processor_parameters=pre_processor_parameters,
                   vectorizer_parameters=vectorizer_parameters,
                   estimator_parameters=estimator_parameters)

    #estimate predictive performance
    #model.estimate( iterable_pos_test, iterable_neg_test, cv=5 )
    # Had to change this call, estimate has no cv parameter
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



from numpy.random import randint
from numpy.random import uniform

pos_iterator=make_iterable(active_fname, 'sdf')
neg_iterator=make_iterable(inactive_fname, 'sdf')

model_fname= 'models/AID%s.model3d_noconfs'%AID

n_iter = 20

pre_processor_parameters={'k':randint(1, 10,size=n_iter),
                          'threshold':randint(1, 10, size=n_iter),
                          'model_type':['3d'],
                          'n_conf':[0]}

logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + 
                " --- Starting model training for AID %s, mode 3d, no conformers" % AID)


model = train_obabel_model(pos_iterator, neg_iterator, pre_processor_parameters,
                           model_type = "3d",
                           model_fname=model_fname,
                           n_iter=n_iter,
                           active_set_size=0,
                           n_active_learning_iterations=0,
                           threshold=10,
                           train_test_split=0.7,
                           verbose=2)

logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + 
                " --- Model training finished.")