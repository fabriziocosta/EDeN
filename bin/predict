#!/usr/bin/env python

import sys
import os
import time

from sklearn.linear_model import SGDClassifier
import numpy as np

from eden import graph
from eden.graphicalizer.graph import node_link_data
from eden.util import setup, eden_io, util

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Predict using model.
"""

def setup_parameters(parser):
	parser = setup.common_arguments(parser)
	parser.add_argument("-m", "--model-dir", 
	dest = "model_dir_path", 
	help = "Path to directory containing the predictive model.",
	default = "out")
	return parser


def predict(args):
	"""
	Predict using model.
	"""
	#load models
	vec = eden_io.load(output_dir_path = args.model_dir_path, out_file_name = "vectorizer")
	logger.info('Vectorizer: %s' % vec)

	clf = eden_io.load(output_dir_path = args.model_dir_path, out_file_name = "model")
	logger.info('Model: %s' % clf)

	#load data
	g_it = node_link_data.node_link_data_to_eden(input = args.input_file, input_type = "file")
	X = vec.transform(g_it, n_jobs = args.n_jobs)
	logger.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))

	#compute predictions using SGD model
	predictions = clf.predict(X)
	margins = clf.decision_function(X)

	#log prediction statistics
	msg = 'Predictions statistics: '
	msg += util.report_base_statistics(predictions)
	logger.info(msg)

	#save results
	#temporary hack
	if args.output_format == "MatrixMarket":
		args.output_format = "text"
		
	out_file_name = "predictions"
	eden_io.store_matrix(matrix = predictions, output_dir_path = args.output_dir_path, out_file_name = out_file_name, output_format = args.output_format)
	logger.info("Written file: %s/%s",args.output_dir_path, out_file_name)

	out_file_name = "margins"
	eden_io.store_matrix(matrix = margins, output_dir_path = args.output_dir_path, out_file_name = out_file_name, output_format = args.output_format)
	logger.info("Written file: %s/%s",args.output_dir_path, out_file_name)


if __name__  == "__main__":
	start_time = time.clock()
	args = setup.arguments_parser(DESCRIPTION, setup_parameters)
	logger = setup.logger(logger_name = "predict", filename = "log", verbosity = args.verbosity)

	logger.info('-'*80)
	logger.info('Program: %s' % sys.argv[0])
	logger.info('Parameters: %s' % args.__dict__)
	try:
		predict(args)
	except Exception:
		import datetime
		curr_time = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
		logger.exception("Program run failed on %s" % curr_time)
	finally:
		end_time = time.clock()
		logger.info('Elapsed time: %.1f sec',end_time - start_time)	