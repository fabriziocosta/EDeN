#!/usr/bin/env python

import sys
import os
import logging

from sklearn.linear_model import SGDClassifier
import numpy as np

from eden import graph
from eden.converters import dispatcher
from eden.util import argument_parser, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Test predictive model.
"""

def setup_parameters(parser):
	parser = argument_parser.setup_common_parameters(parser)
	return parser


def main(args):
	"""
	Test predictive model.
	"""
	#load models
	vec = eden_io.load(output_dir_path = args.output_dir_path, out_file_name = "vectorizer")
	logging.info('Vectorizer: %s' % vec)

	clf = eden_io.load(output_dir_path = args.output_dir_path, out_file_name = "model")
	logging.info('Model: %s' % clf)

	#load data
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)	
	X = vec.transform(g_it, n_jobs = args.n_jobs)
	logging.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))

	#test SGD predicitve model
	predictions = clf.predict(X)
	margins = clf.decision_function(X)
	logging.info('Prediction: %d instances' % (predictions.shape[0]))

	#save results
	#temporary hack
	if args.output_format == "MatrixMarket":
		args.output_format = "text"
	eden_io.store_matrix(matrix = predictions, output_dir_path = args.output_dir_path, out_file_name = "predictions", output_format = args.output_format)
	eden_io.store_matrix(matrix = margins, output_dir_path = args.output_dir_path, out_file_name = "margins", output_format = args.output_format)


if __name__  == "__main__":
	args = setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Program: %s' % sys.argv[0])
	logging.info('Started')
	main(args)
	logging.info('Finished')