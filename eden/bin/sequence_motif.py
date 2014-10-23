#!/usr/bin/env python

import sys
import os
import logging

from sklearn.linear_model import SGDClassifier
import numpy as np

from eden import graph
from eden.converters import dispatcher
from eden.util import argument_parser, setup, eden_io
from eden import iterated_maximum_subarray

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Extract linear motifs by annotating graphs using a predictive model.

Note: the program assumes the existence of an output directory that contains 
a file called 'model' and a file called 'vectorizer'. You can create these files by 
first running the fit.py program on the same dataset.  
"""

def setup_parameters(parser):
	parser.add_argument("-i", "--input-file",
		dest = "input_file",
    	help = "File name with graphs.", 
    	required = True)
	parser.add_argument("-f", "--format",  
		choices = ["gspan", "node_link_data", "obabel", "sequence"],
    	help = "File format.", 
    	default = "sequence")
	parser.add_argument("-o", "--output-dir", 
		dest = "output_dir_path", 
		help = "Path to output directory.",
		default = "out")
	parser.add_argument( "-w", "--reweight-factor",
		dest = "reweight",
		type = float,
		help = """
            Update the 'weight' information as a linear combination of the previuous weight and 
            the absolute value of the margin. 
            If reweight = 0 then do not update.
            If reweight = 1 then discard previous weight information and use only abs(margin)
            If reweight = 0.5 then update with the aritmetic mean of the previous weight information 
            and the abs(margin)
            """,
        default = 1)
	parser.add_argument( "-k", "--min-motif-size",
		dest = "min_motif_size",
		type = int,
		help = "Minimal length of the motif to extract.",
        default = 5)
	parser.add_argument("-v", "--verbosity", 
		action = "count",
		help = "Increase output verbosity")
	return parser


def main(args):
	"""
	Extract linear motifs by annotating graphs using a predictive model.
	"""
	#load vectorizer
	vec = eden_io.load(output_dir_path = args.output_dir_path, out_file_name = "vectorizer")
	logging.info('Vectorizer: %s' % vec)

	#load predictive model
	clf = eden_io.load(output_dir_path = args.output_dir_path, out_file_name = "model")
	logging.info('Model: %s' % clf)

	#initialize annotator
	ann=graph.Annotator(estimator=clf, vectorizer = vec, reweight = args.reweight)
	
	#load data
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)
	
	#annotate
	ann_g_list=[g for g in  ann.transform(g_it)]
	
	#extract_motifs for each graph
	motif_list = []
	for g in ann_g_list:
		motifs = iterated_maximum_subarray.extract_motifs(graph = g, min_motif_size = args.min_motif_size)
		if motifs:
			motif_list += [motifs]
	
	#save results
	full_out_file_name = os.path.join(args.output_dir_path, "motifs.data")
	with open(full_out_file_name, "w") as f:
		for motif in motif_list:
			if "".join(motif[0]):
				line = "motif:%s begin:%d end:%d seq:%s\n" % ("".join(motif[0]),motif[1],motif[2],"".join(motif[3]))
				f.write(line)
	full_out_file_name = os.path.join(args.output_dir_path, "motifs")
	with open(full_out_file_name, "w") as f:
		for motif in motif_list:
			if "".join(motif[0]):
				line = "%s\n" % "".join(motif[0])
				f.write(line)


if __name__  == "__main__":
	args = setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Program: %s' % sys.argv[0])
	logging.info('Started')
	logging.info('Parameters: %s' % args)
	main(args)
	logging.info('Finished')