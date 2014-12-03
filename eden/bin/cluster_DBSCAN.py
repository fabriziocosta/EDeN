#!/usr/bin/env python

import sys
import os
import time

from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
import numpy as np

from eden import graph
from eden.graphicalizer.graph import node_link_data
from eden.util import util, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Clustering with Ward hierarchical clustering: constructs a tree and cuts it.
"""

def setup_parameters(parser):
	parser = setup.common_arguments(parser)
	parser.add_argument("-e","--eps",
		type = float, 
		default = 0.5,
		help = "The maximum distance between two samples for them to be considered as in the same neighborhood.")
	parser.add_argument( "-n","--num-components",
		dest = "n_components",
		type = int, 
		help = "Number of dimensions for PCA approximation.", 
		default = 128)
	return parser


def cluster_DBSCAN(args):
	"""
	Clustering with Ward hierarchical clustering: constructs a tree and cuts it.
	"""
	#load data
	g_it = node_link_data.node_link_data_to_eden(input = args.input_file, input_type = "file")
	vec = graph.Vectorizer(r = args.radius,d = args.distance, nbits = args.nbits)
	logger.info('Vectorizer: %s' % vec)

	X = vec.transform(g_it, n_jobs = args.n_jobs)
	logger.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))
	
	#project to lower dimensional space to use clustering algorithms
	transformer = TruncatedSVD(n_components=args.n_components)
	X_dense=transformer.fit_transform(X)

	#log statistics on data
	logger.info('Dimensionality reduction Instances: %d Features: %d with an avg of %d features per instance' % (X_dense.shape[0], X_dense.shape[1], X.getnnz() / X.shape[0]))

	#clustering
	clustering_algo = DBSCAN(eps = args.eps)
	y = clustering_algo.fit_predict(X_dense)
	msg = 'Predictions statistics: '
	msg += util.report_base_statistics(y)
	logger.info(msg)

	#save model for vectorizer
	out_file_name = "vectorizer"
	eden_io.dump(vec, output_dir_path = args.output_dir_path, out_file_name = out_file_name)
	logger.info("Written file: %s/%s",args.output_dir_path, out_file_name)

	#save result
	out_file_name = "labels"
	eden_io.store_matrix(matrix = y, output_dir_path = args.output_dir_path, out_file_name = out_file_name, output_format = "text")
	logger.info("Written file: %s/%s",args.output_dir_path, out_file_name)
	


if __name__  == "__main__":
	start_time = time.clock()
	args = setup.arguments_parser(DESCRIPTION, setup_parameters)
	logger = setup.logger(logger_name = "cluster_DBSCAN", filename = "log", verbosity = args.verbosity)

	logger.info('-'*80)
	logger.info('Program: %s' % sys.argv[0])
	logger.info('Parameters: %s' % args.__dict__)
	cluster_DBSCAN(args)
	end_time = time.clock()
	logger.info('Elapsed time: %.1f sec',end_time - start_time)