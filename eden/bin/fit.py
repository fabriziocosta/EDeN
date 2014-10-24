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
    	help = """Type of hyper parameter optimization for the classifier. 
    	none:	uses default values for the SGD classifier
    	predictor: optimizes all hyper-parameters of the SGD classifier, such as  the type of penalty, the num of iterations, etc.
    	full: optimizes the vectorizer and the predictor; all radius values between min-r and r are evalauted and selected values of 
    	distance between min-d and d are evaluated.
    	""", 
    	default = "none")
	parser.add_argument("-x", "--output-CV-performance", 
		dest = "output_CV_performance",
		help = "Performe cross validated predictive performace estimation.",
		action = "store_true")
	return parser


def extract_data_matrix(args, vectorizer = None):
	#load data
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)	
	X = vectorizer.transform(g_it, n_jobs = args.n_jobs)
	
	#if data is provided as individual files for positive and negative isntances then join the data matrices and create a corresonding target vector
	if args.neg_file_name != "":
		g_neg_it = dispatcher.any_format_to_eden(input_file = args.neg_file_name, format = args.format)	
		X_neg = vectorizer.transform(g_neg_it, n_jobs = args.n_jobs)
		#create target array	
		yp = [1] * X.shape[0]
		yn = [-1] * X_neg.shape[0]
		y = np.array(yp + yn)
		#update data matrix
		X = vstack( [X,X_neg] , format = "csr")
	else:
		#load target
		y = eden_io.load_target(args.target, input_type = "file")
	#export data
	return X,y


def performace_estimation(predictor = None, data_matrix = None, target = None, cv = 10, scoring = "roc_auc"):
	scores = cross_validation.cross_val_score(predictor, data_matrix, target, cv = cv, scoring = scoring)
	perf = np.mean(scores)
	std = np.std(scores)
	return (perf,std)


def optimize_predictor(predictor = None, data_matrix = None, target = None, n_iter_search = 20, cv = 3, scoring = "roc_auc", n_jobs = -1):
	param_dist = {"n_iter": randint(5, 100),
		"power_t": uniform(0.1),
		"alpha": uniform(1e-08,1e-03),
		"eta0" : uniform(1e-03,10),
		"penalty": ["l1", "l2", "elasticnet"],
		"learning_rate": ["invscaling", "constant","optimal"]}
	optclf = RandomizedSearchCV(predictor, param_distributions = param_dist, n_iter = n_iter_search, cv = cv, scoring = scoring, refit = True, n_jobs = n_jobs)
	optclf.fit(data_matrix, target)
	return optclf.best_estimator_


def optimize_vectorizer(args, predictor = None):
	max_predictor = None
	max_score = 0
	max_vectorizer = None
	#iterate over r
	for r in range(args.min_r,args.radius + 1):
		#iterate over selected d
		for d in set([0,r / 2,r,2 * r]):
			if d >= args.min_d and d <= args.distance:
				vectorizer = graph.Vectorizer(r = r, d = d, min_r = args.min_r, min_d = args.min_d, nbits = args.nbits)
				#load data and extract features
				X,y = extract_data_matrix(args, vectorizer)
				#optimize for predictor
				predictor = optimize_predictor(predictor = predictor, data_matrix = X, target = y, n_jobs = args.n_jobs)	
				score, std = performace_estimation(predictor = predictor, data_matrix = X, target = y)
				#keep max
				if max_score < score :
					max_score = score
					max_predictor = predictor
					max_vectorizer = vectorizer
					logging.info("Increased performance for r: %d   d: %d   score: %.4f (std: %.4f)" % (r, d, score, std))
					#log statistics on data
					logging.info('Target size: %d Target classes: %d' % (y.shape[0],len(set(y))))
					logging.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))
	return max_predictor,vectorizer


def optimize(args, predictor = None):	
	if args.optimization == "none" or args.optimization == "predictor":
		vectorizer = graph.Vectorizer(r = args.radius, d = args.distance, min_r = args.min_r, min_d = args.min_d, nbits = args.nbits)
		#load data and extract features
		X,y = extract_data_matrix(args, vectorizer = vectorizer)
		#log statistics on data
		logging.info('Target size: %d Target classes: %d' % (y.shape[0],len(set(y))))
		logging.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))

	if args.optimization == "none":
		predictor.fit(X,y)
	elif args.optimization == "predictor":
		predictor = optimize_predictor(predictor = predictor, data_matrix = X, target = y, n_jobs = args.n_jobs)	
	elif args.optimization == "full":
		predictor,vectorizer = optimize_vectorizer(args, predictor = predictor)	
		#extract data amtrix for evaluation 
		X,y = extract_data_matrix(args, vectorizer = vectorizer)
		
	score, std = performace_estimation(predictor = predictor, data_matrix = X, target = y)
	logging.info("Predictive score: %.4f (std: %.4f)" % (score, std))

	#save model for vectorizer
	eden_io.dump(vectorizer, output_dir_path = args.output_dir_path, out_file_name = "vectorizer")
	
	return predictor,score,std	

def main(args):
	"""
	Fit predictive model.
	"""
	predictor = SGDClassifier(n_jobs = args.n_jobs, class_weight = 'auto', shuffle = True)
			
	#train and optimize a SGD predicitve model
	predictor,score,std = optimize(args, predictor = predictor)

	#optionally output predictive performance
	if args.output_CV_performance:
		print("CV estimate of AUC ROC predictive performance: %.4f (std: %.4f)" % (score, std))
	
	#save model
	eden_io.dump(predictor, output_dir_path = args.output_dir_path, out_file_name = "model")


if __name__  == "__main__":
	args = setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Program: %s' % sys.argv[0])
	logging.info('Started')
	logging.info('Parameters: %s' % args)
	main(args)
	logging.info('Finished')