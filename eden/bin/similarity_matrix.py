#!/usr/bin/env python

import sys
import os
import argparse
import logging

import numpy as np
from sklearn import metrics

from eden import graph
from eden.converters import gspan,node_link_data,obabel

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Gram matrix computation.
"""
EPILOG="""
Cite:Costa, Fabrizio, and Kurt De Grave. 
Fast neighborhood subgraph pairwise distance kernel. 
Proceedings of the 26th International Conference on Machine Learning. 2010.
"""

def setup_parameters(parser):
	parser.add_argument("-i", "--input-file",  
    	dest = "input_file",
    	help = "File name with graphs.", 
    	required = True)
	parser.add_argument("-f", "--format",  choices = ["gspan", "node_link_data", "obabel"],
    	help = "File format.", 
    	default = "gspan")
	parser.add_argument( "-r","--radius",
		type = int, 
		help = "Size of the largest radius used in EDeN.", 
		default = 2)
	parser.add_argument( "-d", "--distance",
		type = int, 
		help = "Size of the largest distance used in EDeN.", 
		default = 5)
	parser.add_argument("-o", "--output-dir", 
		dest = "output_dir_path", 
		help = "Path to output directory.",
		default = "out")	
	parser.add_argument("-v", "--verbosity", 
		action = "count",
		help = "Increase output verbosity")

def main(args):
	"""
	Gram matrix computation.
	"""
	g_it = gspan.gspan_to_eden(args.input_file, input_type = "file")
	if args.format is "node_link_data":
		g_it = node_link_data.node_link_data_to_eden(args.input_file, input_type = "file")
	if args.format is "obabel":
		g_it = obabel.obabel_to_eden(args.input_file, input_type = "file")
	
	vec = graph.Vectorizer(r = args.radius,d = args.distance)
	X = vec.transform(g_it, n_jobs = None)
	K = metrics.pairwise.pairwise_kernels(X, metric = "linear")
	
	if not os.path.exists(args.output_dir_path) :
		os.mkdir(args.output_dir_path)
	out_file_name  =  "matrix"
	full_out_file_name = os.path.join(args.output_dir_path, out_file_name)
	np.savetxt(full_out_file_name,K, fmt = '%.4f')


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