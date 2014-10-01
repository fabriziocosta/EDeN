#!/usr/bin/env python

import sys
import os
import logging

from sklearn.cluster import DBSCAN
from sklearn.decomposition import RandomizedPCA
import numpy as np

from eden import graph
from eden.converters import dispatcher
from eden.util import argument_parser, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Clustering with Ward hierarchical clustering: constructs a tree and cuts it.
"""

def setup_parameters(parser):
	parser = argument_parser.setup_common_parameters(parser)
	parser.add_argument("-e","--eps",
		type = float, 
		default = 0.5,
		help = "The maximum distance between two samples for them to be considered as in the same neighborhood.")
	parser.add_argument( "-g","--num-components",
		dest = "num_components",
		type = int, 
		help = "Number of dimensions for PCA approximation.", 
		default = 256)
	return parser


def main(args):
	"""
	Clustering with Ward hierarchical clustering: constructs a tree and cuts it.
	"""
	#load data
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)	
	vec = graph.Vectorizer(r = args.radius,d = args.distance, nbits = args.nbits)
	X = vec.transform(g_it, n_jobs = args.n_jobs)
	
	#log statistics on data
	logging.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))

	#project to lower dimensional space to use clustering algorithms
	pca = RandomizedPCA(n_components=args.num_components)
	X_dense=pca.fit_transform(X)

	#clustering
	clustering_algo = DBSCAN(eps = args.eps)
	y = clustering_algo.fit_predict(X_dense) 

	#save model
	eden_io.dump(vec, output_dir_path = args.output_dir_path, out_file_name = "vectorizer")
	eden_io.store_matrix(matrix = y, output_dir_path = args.output_dir_path, out_file_name = "labels", output_format = "text")


if __name__  == "__main__":
	args = setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Program: %s' % sys.argv[0])
	logging.info('Started')
	main(args)
	logging.info('Finished')