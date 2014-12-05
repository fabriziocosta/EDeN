#!/usr/bin/env python

import sys
import os
import time

from sklearn.neighbors import kneighbors_graph

from eden import graph
from eden.graphicalizer.graph import node_link_data
from eden.util import util, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Nearest neighbor computation.
"""

def setup_parameters(parser):
	parser = setup.common_arguments(parser)
	parser.add_argument( "-k","--num-nearest-neighbors",
		dest = "num_neighbours",
		type = int, 
		help = "Number of nearest neighbors to compute.", 
		default = 3)
	parser.add_argument("-m", "--mode",  choices = ["connectivity", "distance"],
    	help = "Type of returned matrix: 'connectivity' will return the connectivity matrix with ones and zeros, in 'distance' the edges are Euclidean distance between points.", 
    	default = "distance")
	return parser


def nearest_neighbors(args):
	"""
	Nearest neighbor computation.
	"""
	g_it = node_link_data.node_link_data_to_eden(input = args.input_file, input_type = "file")
	vec = graph.Vectorizer(r = args.radius,d = args.distance, nbits = args.nbits)
	logger.info('Vectorizer: %s' % vec)

	X = vec.transform(g_it, n_jobs = args.n_jobs)
	logger.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz()/X.shape[0]))

	A = kneighbors_graph(X, args.num_neighbours, mode = args.mode)
	
	#output
	out_file_name = "matrix"	
	eden_io.store_matrix(matrix = A, output_dir_path = args.output_dir_path, out_file_name = out_file_name, output_format = args.output_format)
	logger.info("Written file: %s/%s",args.output_dir_path, out_file_name)



if __name__  == "__main__":
	start_time = time.clock()
	args = setup.arguments_parser(DESCRIPTION, setup_parameters)
	logger = setup.logger(logger_name = "nearest_neighbors", filename = "log", verbosity = args.verbosity)

	logger.info('-'*80)
	logger.info('Program: %s' % sys.argv[0])
	logger.info('Parameters: %s' % args.__dict__)
	try:
		nearest_neighbors(args)
	except Exception:
		import datetime
		curr_time = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
		logger.exception("Program run failed on %s" % curr_time)
	finally:
		end_time = time.clock()
		logger.info('Elapsed time: %.1f sec',end_time - start_time)	