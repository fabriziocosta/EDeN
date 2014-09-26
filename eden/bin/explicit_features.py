#!/usr/bin/env python

import sys
import os
import logging

from sklearn import metrics

import numpy as np
from scipy import io
from sklearn.externals import joblib

from eden import graph
from eden.converters import gspan,node_link_data,obabel


from eden.util import argument_parser
from eden.util import logging_setup
from eden.util import io as eden_io


DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Explicit features computation.
"""

def setup_parameters(parser):
	argument_parser.setup_common_parameters(parser)
	

def main(args):
	"""
	Explicit features computation.
	"""
	g_it = gspan.gspan_to_eden(args.input_file, input_type = "file")
	if args.format is "node_link_data":
		g_it = node_link_data.node_link_data_to_eden(args.input_file, input_type = "file")
	if args.format is "obabel":
		g_it = obabel.obabel_to_eden(args.input_file, input_type = "file")
	vec = graph.Vectorizer(r = args.radius,d = args.distance, nbits = args.nbits)
	if args.n_jobs is -1:
		n_jobs = None
	else:
		n_jobs = args.n_jobs
	X = vec.transform(g_it, n_jobs = n_jobs)
	logging.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz()/X.shape[0]))
	eden_io.output_matrix(output_dir_path=args.output_dir_path, 
		out_file_name="features", 
		output_format=args.output_format, 
		matrix=X)


if __name__  == "__main__":
	args=logging_setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Started')
	main(args)
	logging.info('Finished')