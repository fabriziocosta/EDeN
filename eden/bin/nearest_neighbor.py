#!/usr/bin/env python

import sys
import os
import argparse
import logging

import numpy as np
from scipy import io
from sklearn.neighbors import kneighbors_graph
from sklearn.externals import joblib

from eden import graph
from eden.converters import gspan,node_link_data,obabel

from eden.util.globals import EPILOG
from eden.util import argument_parser

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Nearest neighbor computation.
"""

def setup_parameters(parser):
	argument_parser.setup_common_parameters(parser)
	parser.add_argument( "-k","--num-nearest-neighbors",
		dest = "num_neighbours",
		type = int, 
		help = "Number of nearest neighbors to compute.", 
		default = 3)
	parser.add_argument("-m", "--mode",  choices = ["connectivity", "distance"],
    	help = "Type of returned matrix: 'connectivity' will return the connectivity matrix with ones and zeros, in 'distance' the edges are Euclidean distance between points.", 
    	default = "distance")


def main(args):
	"""
	Nearest neighbor computation.
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
	A = kneighbors_graph(X, args.num_neighbours, mode = args.mode)
	
	if not os.path.exists(args.output_dir_path) :
		os.mkdir(args.output_dir_path)
	out_file_name  =  "neighbors"
	full_out_file_name = os.path.join(args.output_dir_path, out_file_name)
	if args.output_format == "MatrixMarket":
		io.mmwrite(full_out_file_name, A, precision = 4)
	elif args.output_format == "numpy":
		np.save(full_out_file_name, A)
	elif args.output_format == "joblib":
		joblib.dump(A, full_out_file_name) 

if __name__  == "__main__":
	parser = argparse.ArgumentParser(description=DESCRIPTION,
		epilog=EPILOG,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	setup_parameters(parser)
	args = parser.parse_args()

	log_level = logging.INFO
	if args.verbosity == 1:
		log_level = logging.WARNING
		print "WARNING"
	elif args.verbosity >= 2:
		log_level = logging.DEBUG
		print "DEBUG"
	logging.basicConfig(level=log_level)

	main(args)