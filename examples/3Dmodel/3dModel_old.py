
# coding: utf-8

# In[1]:

from eden.converter.molecule import obabel
import networkx as nx
import pybel
import requests
import os.path


# In[2]:

AID=825
READ_FROM_FILE=True
DATA_DIR = '/Users/jl/uni-freiburg/thesis/EDeN/examples/3Dmodel/data'
active_fname=DATA_DIR + '/AID%s_active.sdf'%AID
inactive_fname=DATA_DIR + '/AID%s_inactive.sdf'%AID


# In[3]:

### Begin train function
import datetime
pos_fname = active_fname
neg_fname = inactive_fname
model_type = "default"
train_test_split = .5
n_iter = 5
model_fname=DATA_DIR + '/AID%s.model3d'%AID

from numpy.random import randint
from numpy.random import uniform

# this will be passed as an argument to the model later on
def pre_processor( data, **args):
    print "yay"
    return data

from eden.graph import Vectorizer
#vectorizer_parameters={'complexity':[1,2,3,4],
#                       'discretization_size':randint(3, 100,size=n_iter),
#                       'discretization_dimension':randint(3, 50, size=n_iter)}
vectorizer_parameters={'complexity':[1,2,3,4],
                       'discretization_size':3,
                       'discretization_dimension':3}
vectorizer = Vectorizer(**vectorizer_parameters)

from sklearn.linear_model import SGDClassifier
estimator = SGDClassifier(class_weight='auto', shuffle=True)


# In[9]:

# Create iterable from files

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

iterable_pos, iterable_pos_ = tee(iterable_pos)
iterable_neg, iterable_neg_ = tee(iterable_neg)
#vectorizer.fit(iterable_pos_)

# At this point iterable_pos contains the networkx graph objects generated by the converter
#for i in iterable_pos:
#    print type(i)


# In[10]:

#split train/test
from eden.util import random_bipartition_iter
iterable_pos_train, iterable_pos_test = random_bipartition_iter(iterable_pos, relative_size=train_test_split)
iterable_neg_train, iterable_neg_test = random_bipartition_iter(iterable_neg, relative_size=train_test_split)



#make predictive model
from eden.model import ActiveLearningBinaryClassificationModel
model = ActiveLearningBinaryClassificationModel( pre_processor, estimator=estimator, vectorizer=vectorizer )

#optimize hyperparameters and fit model

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

#print "*"*40
#print "Type of iterable_pos_train: ", str(type(iterable_pos_train))
#for i in iterable_pos_train:
#    print str(type(i))

### The elements of iterable_pos_train are still networkx graph objects


# In[11]:


model.optimize(iterable_pos_train, iterable_neg_train,
               model_name=model_fname,
               fit_vectorizer=True,
               n_active_learning_iterations=0,
               size_positive=-1,
               size_negative=500,
               n_iter=n_iter, cv=3, n_jobs=1, verbose=1,
               pre_processor_parameters=pre_processor_parameters,
               vectorizer_parameters=vectorizer_parameters,
               estimator_parameters=estimator_parameters)

#estimate predictive performance
#model.estimate( iterable_pos_test, iterable_neg_test, cv=5 )
# Had to change this call, estimate has no cv parameter
model.estimate( iterable_pos_test, iterable_neg_test )