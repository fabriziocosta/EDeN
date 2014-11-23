#!/usr/bin/env python

import sys
import os
import time
import logging
import logging.handlers

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


def nearest_neighbors(args):
	"""
	Nearest neighbor computation.
	"""
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)	
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
	args = setup.setup(DESCRIPTION, setup_parameters)

	logger = logging.getLogger("nearest_neighbors")
	log_level = logging.WARNING
	if args.verbosity == 1:
		log_level = logging.INFO
	elif args.verbosity >= 2:
		log_level = logging.DEBUG
	logger.setLevel(logging.DEBUG)
	# create console handler
	ch = logging.StreamHandler()
	ch.setLevel(log_level)
	# create a file handler
	fh = logging.handlers.RotatingFileHandler(filename = "log" , maxBytes=100000, backupCount=10)
	fh.setLevel(logging.DEBUG)
	# create formatter
	cformatter = logging.Formatter('%(message)s')
	# add formatter to ch
	ch.setFormatter(cformatter)
	# create formatter
	fformatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# and to fh
	fh.setFormatter(fformatter)
	# add handlers to logger
	logger.addHandler(ch)
	logger.addHandler(fh)

	logger.info('-------------------------------------------------------')
	logger.info('Program: %s' % sys.argv[0])
	logger.info('Parameters: %s' % args)
	nearest_neighbors(args)
	end_time = time.clock()
	logger.info('Elapsed time: %.1f sec',end_time - start_time)