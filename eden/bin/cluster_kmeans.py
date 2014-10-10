#!/usr/bin/env python

import sys
import os
import logging

from sklearn.cluster import MiniBatchKMeans
import numpy as np

from eden import graph
from eden.converters import dispatcher
from eden.util import argument_parser, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Clustering with mini-batch K-Means.
"""

def setup_parameters(parser):
	parser = argument_parser.setup_common_parameters(parser)
	parser.add_argument("-y","--num-clusters",
		dest = "n_clusters",
		type = int, 
		default = 8,
		help = "Number of expected clusters.")
	parser.add_argument("-n","--num-random-initializations",
		dest = "n_init",
		type = int, 
		default = 3,
		help = """Number of random initializations.""")
	return parser


def main(args):
	"""
	Clustering with mini-batch K-Means.
	"""
	#load data
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)	
	vec = graph.Vectorizer(r = args.radius,d = args.distance, nbits = args.nbits)
	X = vec.transform(g_it, n_jobs = args.n_jobs)
	
	#log statistics on data
	logging.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))

	#clustering
	clustering_algo = MiniBatchKMeans(n_clusters = args.n_clusters, n_init = args.n_init)
	y = clustering_algo.fit_predict(X) 

	#save model
	eden_io.dump(vec, output_dir_path = args.output_dir_path, out_file_name = "vectorizer")
	eden_io.store_matrix(matrix = y, output_dir_path = args.output_dir_path, out_file_name = "labels", output_format = "text")


if __name__  == "__main__":
	args = setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Program: %s' % sys.argv[0])
	logging.info('Started')
	main(args)
	logging.info('Finished')
