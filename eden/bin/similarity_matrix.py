#!/usr/bin/env python

import sys
import os
import logging

from sklearn import metrics

from eden import graph
from eden.converters import dispatcher
from eden.util import argument_parser, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Gram matrix computation.
"""

def setup_parameters(parser):
	return argument_parser.setup_common_parameters(parser)


def main(args):
	"""
	Gram matrix computation.
	"""
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)	
	vec = graph.Vectorizer(r = args.radius,d = args.distance, nbits = args.nbits)
	X = vec.transform(g_it, n_jobs = args.n_jobs)
	logging.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz()/X.shape[0]))

	K = metrics.pairwise.pairwise_kernels(X, metric = "linear", n_jobs = args.n_jobs)
	
	#output
	eden_io.store_matrix(matrix = K, output_dir_path = args.output_dir_path, out_file_name = "matrix", output_format = args.output_format)


if __name__  == "__main__":
	args = setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Program: %s' % sys.argv[0])
	logging.info('Started')
	main(args)
	logging.info('Finished')