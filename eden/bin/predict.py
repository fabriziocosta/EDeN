#!/usr/bin/env python

import sys
import os
import time
import logging
import logging.handlers

from sklearn.linear_model import SGDClassifier
import numpy as np

from eden import graph
from eden.converters import dispatcher
from eden.util import argument_parser, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Predict using model.
"""

def setup_parameters(parser):
	parser = argument_parser.setup_common_parameters(parser)
	return parser


def report_base_statistics(vec):
	from collections import Counter
	c =Counter(vec)
	msg = ''
	for k in c:
   		msg += "class: %s count:%d (%0.2f)\t"% (k, c[k], c[k]/float(len(vec)))
   	return msg


def predict(args):
	"""
	Predict using model.
	"""
	#load models
	vec = eden_io.load(output_dir_path = args.output_dir_path, out_file_name = "vectorizer")
	logger.info('Vectorizer: %s' % vec)

	clf = eden_io.load(output_dir_path = args.output_dir_path, out_file_name = "model")
	logger.info('Model: %s' % clf)

	#load data
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)	
	X = vec.transform(g_it, n_jobs = args.n_jobs)
	logger.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))

	#compute predictions using SGD model
	predictions = clf.predict(X)
	margins = clf.decision_function(X)

	#log prediction statistics
	msg = 'Predictions statistics: '
	msg += report_base_statistics(predictions)
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
	args = setup.setup(DESCRIPTION, setup_parameters)

	logger = logging.getLogger("predict")
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
	predict(args)
	end_time = time.clock()
	logger.info('Elapsed time: %.1f sec',end_time - start_time)