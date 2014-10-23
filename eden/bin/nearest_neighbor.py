#!/usr/bin/env python

import sys
import os
import logging

from sklearn.neighbors import kneighbors_graph

from eden import graph
from eden.converters import dispatcher
from eden.util import argument_parser, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Nearest neighbor computation.
"""

def setup_parameters(parser):
	parser = argument_parser.setup_common_parameters(parser)
	parser.add_argument( "-k","--num-nearest-neighbors",
		dest = "num_neighbours",
		type = int, 
		help = "Number of nearest neighbors to compute.", 
		default = 3)
	parser.add_argument("-m", "--mode",  choices = ["connectivity", "distance"],
    	help = "Type of returned matrix: 'connectivity' will return the connectivity matrix with ones and zeros, in 'distance' the edges are Euclidean distance between points.", 
    	default = "distance")
	return parser


def main(args):
	"""
	Nearest neighbor computation.
	"""
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)	
	vec = graph.Vectorizer(r = args.radius,d = args.distance, nbits = args.nbits)
	X = vec.transform(g_it, n_jobs = args.n_jobs)
	logging.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz()/X.shape[0]))
	A = kneighbors_graph(X, args.num_neighbours, mode = args.mode)
	
	#output
	eden_io.store_matrix(matrix = A, output_dir_path = args.output_dir_path, out_file_name = "matrix", output_format = args.output_format)

if __name__  == "__main__":
	args = setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Program: %s' % sys.argv[0])
	logging.info('Started')
	logging.info('Parameters: %s' % args)
	main(args)
	logging.info('Finished')