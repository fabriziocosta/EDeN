#!/usr/bin/env python

import sys
import os
import logging

from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn import cross_validation
from scipy.stats import randint
from scipy.stats import uniform
import numpy as np

from eden import graph
from eden.converters import dispatcher
from eden.util import argument_parser, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Train predictive model.
"""

def setup_parameters(parser):
	parser = argument_parser.setup_common_parameters(parser)
	parser.add_argument( "-y","--target-file-name",
		dest = "target",
		help = "Target file. One row per instance.", 
		required = True)
	parser.add_argument("-p", "--optimization",  choices = ["none", "predictor", "full"],
    	help = "Type of hyper parameter optimization for the classifier.", 
    	default = "none")
	parser.add_argument("--output-CV-performance", 
		dest = "output_CV_performance",
		help = "Performe cross validated predictive performace estimation.",
		action = "store_true")
	return parser


def performace_estimation(clf, X, y, cv = 10, scoring = "roc_auc"):
	scores = cross_validation.cross_val_score(clf, X, y, cv = cv, scoring = scoring)
	perf = np.mean(scores)
	std = np.std(scores)
	return (perf,std)


def optimize(clf, X, y, n_iter_search = 20, cv = 3, scoring = "roc_auc", n_jobs = -1):
	param_dist = {"n_iter": randint(5, 100),
		"power_t": uniform(0.1),
		"alpha": uniform(1e-08,1e-03),
		"eta0" : uniform(1e-03,10),
		"penalty": ["l1", "l2", "elasticnet"],
		"learning_rate": ["invscaling", "constant","optimal"]}
	random_search = RandomizedSearchCV(clf, param_distributions = param_dist, n_iter = n_iter_search, cv = cv, scoring = scoring, n_jobs = n_jobs)
	random_search.fit(X, y)
	optclf = SGDClassifier(**random_search.best_params_)
	logging.info("Best parameter configuration")
	logging.info(random_search.best_params_)
	return optclf


def main(args):
	"""
	Train predictive model.
	"""
	#load data
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)	
	vec = graph.Vectorizer(r = args.radius,d = args.distance, nbits = args.nbits)
	X = vec.transform(g_it, n_jobs = args.n_jobs)
	logging.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))

	#load target
	y = eden_io.load_target(args.target, input_type = "file")
	logging.info('Target size: %d Target classes: %d' % (y.shape[0],len(set(y))))

	#train and optimize a SGD predicitve model
	clf = SGDClassifier(n_jobs=args.n_jobs)
	if args.optimization == "none":
		clf.fit(X,y)
	elif args.optimization == "predictor":
		clf = optimize(clf, X, y, n_jobs = args.n_jobs)	
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
	main(args)
	logging.info('Finished')