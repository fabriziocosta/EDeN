#!/usr/bin/env python

import sys
import os
import time

from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn import cross_validation
import sklearn.metrics 
from scipy.stats import randint
from scipy.stats import uniform
from scipy.sparse import vstack
import numpy as np

from eden.util import setup

from eden import graph
from eden.converters import dispatcher
from eden.util import eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Fit predictive model.
"""

def setup_parameters(parser):
	parser = setup.common_arguments(parser)
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
	parser.add_argument("-p", "--optimization",  choices = ["none", "predictor", "predictor_and_vectorizer"],
    	help = """Type of hyper parameter optimization for the classifier. 
    	[none]:	uses default values for the SGD classifier;
    	[predictor]:	optimize all hyper-parameters of the SGD classifier: "penalty" chosen among 
    	["l1", "l2", "elasticnet"], "learning_rate" chosen among  ["invscaling", "constant","optimal"], 
    	"n_iter" in the interval (5, 100), "power_t" in the interval (0.1),	"alpha" in the interval (1e-08,1e-03),
		"eta0" in the interval (1e-03,10);
    	[predictor_and_vectorizer]:	jointly optimize the vectorizer and the predictor; 
    	namely all radius values between min-r and r and up to 4 values of 
    	distance between min-d and d are evaluated, specifically d = [0, r/2, r, 2 * r].
    	""", 
    	default = "predictor_and_vectorizer")
	parser.add_argument("-x", "--output-predictive-performance", 
		dest = "output_predictive_performance",
		help = "Print predictive performace estimation on a 10 cross validation.",
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


def optimize_predictor_and_vectorizer(args, predictor = None):
	max_predictor = None
	max_score = 0
	max_vectorizer = None
	#iterate over r
	for r in range(args.min_r,args.radius + 1):
		#iterate over selected d
		for d in set([0, r / 2, r, 2 * r]):
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
					logger.info("Increased performance for r: %d   d: %d   score: %.4f (std: %.4f)" % (r, d, score, std))
					#log statistics on data
					logger.info('Target size: %d Target classes: %d' % (y.shape[0],len(set(y))))
					logger.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))
	return max_predictor,vectorizer


def report_base_statistics(vec):
	from collections import Counter
	c =Counter(vec)
	msg = ''
	for k in c:
   		msg += "class: %s count:%d (%0.2f)\t"% (k, c[k], c[k]/float(len(vec)))
   	return msg


def optimize(args, predictor = None):	
	if args.optimization == "none" or args.optimization == "predictor":
		vectorizer = graph.Vectorizer(r = args.radius, d = args.distance, min_r = args.min_r, min_d = args.min_d, nbits = args.nbits)
		#load data and extract features
		X,y = extract_data_matrix(args, vectorizer = vectorizer)
		#log statistics on data
		logger.info('Target size: %d Target classes: %d' % (y.shape[0],len(set(y))))
		logger.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))

	if args.optimization == "none":
		predictor.fit(X,y)
	elif args.optimization == "predictor":
		predictor = optimize_predictor(predictor = predictor, data_matrix = X, target = y, n_jobs = args.n_jobs)	
	elif args.optimization == "predictor_and_vectorizer":
		predictor,vectorizer = optimize_predictor_and_vectorizer(args, predictor = predictor)	
		#extract data amtrix for evaluation 
		X,y = extract_data_matrix(args, vectorizer = vectorizer)

	if args.output_predictive_performance:
		msg = 'Target statistics: '
		msg += report_base_statistics(y)
		logger.debug(msg)
		print(msg)
		for scoring in ['accuracy','average_precision','f1','precision','recall','roc_auc']:
			score, std = performace_estimation(predictor = predictor, data_matrix = X, target = y, scoring = scoring)
			msg = "Metric: {0:>17s}: {1:5.4f} (std: {2:5.4f})".format(scoring, score, std)
			logger.debug(msg)
			print(msg)
	else:
		score = None
		std = None
		
	logger.info('Vectorizer: %s' % vectorizer)

	#save model for vectorizer
	out_file_name = "vectorizer"
	eden_io.dump(vectorizer, output_dir_path = args.output_dir_path, out_file_name = out_file_name)
	logger.info("Written file: %s/%s",args.output_dir_path, out_file_name)

	return predictor,score,std	


def fit(args):
	"""
	Fit predictive model.
	"""
	predictor = SGDClassifier(n_jobs = args.n_jobs, class_weight = 'auto', shuffle = True)
			
	#train and optimize a SGD predicitve model
	predictor,score,std = optimize(args, predictor = predictor)

	#save model
	out_file_name = "model"
	eden_io.dump(predictor, output_dir_path = args.output_dir_path, out_file_name = out_file_name)
	logger.info("Written file: %s/%s",args.output_dir_path, out_file_name)



if __name__  == "__main__":
	start_time = time.clock()
	args = setup.arguments_parser(DESCRIPTION, setup_parameters)
	logger = setup.logger(logger_name = "fit", filename = "log", verbosity = args.verbosity)

	logger.info('-'*80)
	logger.info('Program: %s' % sys.argv[0])
	logger.info('Parameters: %s' % args.__dict__)
	fit(args)
	end_time = time.clock()
	logger.info('Elapsed time: %.1f sec',end_time - start_time)