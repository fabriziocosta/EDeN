#!/usr/bin/env python

import sys
import os
import logging

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import metrics
import numpy as np
import pylab as pl

from eden import graph
from eden.converters import dispatcher
from eden.util import argument_parser, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Generate dendrogram plot.
"""

def setup_parameters(parser):
	parser = argument_parser.setup_common_parameters(parser)
	parser.add_argument("-a","--annotation-file",
		dest = "annotation_file",
		required = True,
		help = "File containing instance labels")
	parser.add_argument("--plot-size",
		dest = "plot_size",
		type = int, 
		default = 120,
		help = "Size of the plot area.")
	parser.add_argument("--color-threshold",
		dest = "color_threshold",
		type = float, 
		default = 4,
		help = "Distance threshold for changing color in plot.")
	parser.add_argument("-l", "--linkage",  choices = ["ward","complete","average"],
    	help = """Which linkage criterion to use. 
    	The linkage criterion determines which distance to use between sets of observation. 
    	The algorithm will merge the pairs of cluster that minimize this criterion.
        ward minimizes the variance of the clusters being merged.
        average uses the average of the distances of each observation of the two sets.
        complete or maximum linkage uses the maximum distances between all observations of the two sets.
    	""", 
    	default = "average")
	return parser


def main(args):
	"""
	Generate dendrogram plot.
	"""
	#load data
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)	
	vec = graph.Vectorizer(r = args.radius,d = args.distance, nbits = args.nbits)
	X = vec.transform(g_it, n_jobs = args.n_jobs)
	
	#log statistics on data
	logging.info('Instances: %d Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1], X.getnnz() / X.shape[0]))
	
	with open(args.annotation_file,'r') as f:
		annotations = ['%d:%s' % (num,y.strip()) for num,y in enumerate(f)]

	#compute distance matrix
 	D = metrics.pairwise.pairwise_distances(X, metric = 'cosine')
	
	#compute dendrogram
	fig = pl.figure(figsize = (args.plot_size//10,args.plot_size))
	dendrogram(linkage(D, method = args.linkage), 
		color_threshold = args.color_threshold,
		labels = annotations, 
		orientation = 'right')

	#plot dendrogram
#	ax = fig.add_subplot(111)
#	ax.plot([1,2,3])

	#save plot
	out_file_name = 'dendrogram.pdf'
	if not os.path.exists(args.output_dir_path) :
		os.mkdir(args.output_dir_path)
	full_out_file_name = os.path.join(args.output_dir_path, out_file_name)
	fig.savefig(full_out_file_name)


if __name__  == "__main__":
	args = setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Program: %s' % sys.argv[0])
	logging.info('Started')
	main(args)
	logging.info('Finished')




