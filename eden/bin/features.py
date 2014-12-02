#!/usr/bin/env python

import sys
import os
import time

from eden import graph
from eden.converters import dispatcher
from eden.util import util, setup, eden_io


DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Explicit features computation.
"""

def setup_parameters(parser):
	parser = setup.common_arguments(parser)
	return parser


def features(args):
	"""
	Explicit features computation.
	"""
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)	
	vec = graph.Vectorizer(r = args.radius,d = args.distance, nbits = args.nbits)
	logger.info('Vectorizer: %s' % vec)

	X = vec.transform(g_it, n_jobs = args.n_jobs)
	logger.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz()/X.shape[0]))
	
	#output
	out_file_name = "features"
	eden_io.store_matrix(matrix = X, output_dir_path = args.output_dir_path, out_file_name = out_file_name, output_format = args.output_format)
	logger.info("Written file: %s/%s",args.output_dir_path, out_file_name)


if __name__  == "__main__":
	start_time = time.clock()
	args = setup.arguments_parser(DESCRIPTION, setup_parameters)
	logger = setup.logger(logger_name = "features", filename = "log", verbosity = args.verbosity)

	logger.info('-'*80)
	logger.info('Program: %s' % sys.argv[0])
	logger.info('Parameters: %s' % args.__dict__)
	features(args)
	end_time = time.clock()
	logger.info('Elapsed time: %.1f sec',end_time - start_time)