#!/usr/bin/env python

import sys
import os
import logging

from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn import cross_validation
from scipy.stats import randint
from scipy.stats import uniform
from scipy.sparse import vstack
import numpy as np

from eden import graph
from eden.converters import dispatcher
from eden.util import argument_parser, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Fit predictive model.
"""

def setup_parameters(parser):
	parser = argument_parser.setup_common_parameters(parser)
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument( "-y","--target-file-name",
		dest = "target",
		help = "Target file. One row per instance.")
	group.add_argument( "-n","--neg-file-name",
		dest = "neg_file_name",
		help = """File name for negative instances. 
		If set then  --input-file is assumed containing positive instance.
		This option is mutually exclusive with the --target-file-name option.""",
		default = "")
	parser.add_argument("-p", "--optimization",  choices = ["none", "predictor", "full"],
    	help = "Type of hyper parameter optimization for the classifier.", 
    	default = "none")
	parser.add_argument("-x", "--output-CV-performance", 
		dest = "output_CV_performance",
		help = "Performe cross validated predictive performace estimation.",
		action = "store_true")
	return parser


def performace_estimation(clf, X, y, cv = 10, scoring = "roc_auc"):
	scores = cross_validation.cross_val_score(clf, X, y, cv = cv, scoring = scoring)
	perf = np.mean(scores)
	std = np.std(scores)
	return (perf,std)


def optimize_predictor(clf, X, y, n_iter_search = 20, cv = 3, scoring = "roc_auc", n_jobs = -1):
	param_dist = {"n_iter": randint(5, 100),
		"power_t": uniform(0.1),
		"alpha": uniform(1e-08,1e-03),
		"eta0" : uniform(1e-03,10),
		"penalty": ["l1", "l2", "elasticnet"],
		"learning_rate": ["invscaling", "constant","optimal"]}
	optclf = RandomizedSearchCV(clf, param_distributions = param_dist, 
		n_iter = n_iter_search, 
		cv = cv, 
		scoring = scoring, 
		refit = True,
		n_jobs = n_jobs)
	optclf.fit(X, y)
	logging.info("Best parameter configuration")
	logging.info(optclf.best_params_)
	return optclf.best_estimator_


def main(args):
	"""
	Fit predictive model.
	"""
	#load data
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)	
	vec = graph.Vectorizer(r = args.radius, d = args.distance, min_r = args.min_r, min_d = args.min_d ,nbits = args.nbits)
	X = vec.transform(g_it, n_jobs = args.n_jobs)
	
	#if data is provided as individual files for positive and negative isntances then join the data matrices and create a corresonding target vector
	if args.neg_file_name != "":
		g_neg_it = dispatcher.any_format_to_eden(input_file = args.neg_file_name, format = args.format)	
		X_neg = vec.transform(g_neg_it, n_jobs = args.n_jobs)
		#create target array	
		yp=[1]*X.shape[0]
		yn=[-1]*X_neg.shape[0]
		y=np.array(yp+yn)
		#update data matrix
		X = vstack( [X,X_neg] , format = "csr")
	else:
		#load target
		y = eden_io.load_target(args.target, input_type = "file")

	#log statistics on data
	logging.info('Target size: %d Target classes: %d' % (y.shape[0],len(set(y))))
	logging.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))

	#train and optimize a SGD predicitve model
	clf = SGDClassifier(n_jobs = args.n_jobs, class_weight = 'auto', shuffle = True)
	if args.optimization == "none":
		clf.fit(X,y)
	elif args.optimization == "predictor":
		clf = optimize_predictor(clf, X, y, n_jobs = args.n_jobs)	
	elif args.optimization == "full":
		#TODO: run through r and d
		#log partial runs
		raise Exception("full optimization is not supproted yet")

	#save model
	eden_io.dump(vec, output_dir_path = args.output_dir_path, out_file_name = "vectorizer")
	eden_io.dump(clf, output_dir_path = args.output_dir_path, out_file_name = "model")

	#optionally output predictive performance
	if args.output_CV_performance:
		perf, std = performace_estimation(clf, X, y)
		results = "CV estimate of AUC ROC predictive performance: %.4f (std: %.4f)" % (perf, std)
		print(results)
		logging.info(results)


if __name__  == "__main__":
	args = setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Program: %s' % sys.argv[0])
	logging.info('Started')
	logging.info('Parameters: %s' % args)
	main(args)
	logging.info('Finished')