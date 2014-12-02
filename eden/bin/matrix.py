#!/usr/bin/env python

import sys
import os
import time

from sklearn import metrics

from eden import graph
from eden.converters import dispatcher
from eden.util import util, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Pairwise matrix computation.
"""

def setup_parameters(parser):
	parser = setup.common_arguments(parser)
    # mode is not needed, or?
	parser.add_argument("-m", "--mode",  choices = ["similarity","distance"],
    	help = "switch between pairwise similarity or distance computation.", 
    	default = "similarity")
	parser.add_argument("--distance-metric",  choices = ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan"],
    	dest = "distance_metric",
    	help = """Note that in the case of 'cityblock', 'cosine' and 'euclidean' 
    	(which are valid scipy.spatial.distance metrics), the scikit-learn implementation 
    	will be used, which is faster and has support for sparse matrices 
    	(except for 'cityblock').""", 
    	default = "euclidean")
	parser.add_argument("--similarity-metric",  choices = ["rbf", "sigmoid", "polynomial", "poly", "linear", "cosine"],
    	dest = "similarity_metric",
    	help = """Valid metrics.""", 
    	default = "linear")
	return parser

#TODO:support for kernel parameters from arguments

def matrix(args):
	"""
	Pairwise matrix computation.
	"""

	#input data
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)
	vec = graph.Vectorizer(r = args.radius,d = args.distance, nbits = args.nbits)
	logger.info('Vectorizer: %s' % vec)

	X = vec.transform(g_it, n_jobs = args.n_jobs)
	logger.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz()/X.shape[0]))

	#compute pairwise matrix
	if args.mode == "similarity":
		K = metrics.pairwise.pairwise_kernels(X, metric = args.similarity_metric, n_jobs = args.n_jobs)
	elif args.mode == "distance":
		K = metrics.pairwise.pairwise_distances(X, metric = args.distance_metric, n_jobs = args.n_jobs)
	
	#output
	out_file_name = "matrix"
	eden_io.store_matrix(matrix = K, output_dir_path = args.output_dir_path, out_file_name = out_file_name, output_format = args.output_format)
	logger.info("Written file: %s/%s",args.output_dir_path, out_file_name)


if __name__  == "__main__":
	start_time = time.clock()
	args = setup.arguments_parser(DESCRIPTION, setup_parameters)
	logger = setup.logger(logger_name = "matrix", filename = "log", verbosity = args.verbosity)

	logger.info('-'*80)
	logger.info('Program: %s' % sys.argv[0])
	logger.info('Parameters: %s' % args.__dict__)
	matrix(args)
	end_time = time.clock()
	logger.info('Elapsed time: %.1f sec',end_time - start_time)