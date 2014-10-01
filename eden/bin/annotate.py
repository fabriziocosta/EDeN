#!/usr/bin/env python

import sys
import os
import logging

from sklearn.linear_model import SGDClassifier
import numpy as np

from eden import graph
from eden.converters import dispatcher, node_link_data
from eden.util import argument_parser, setup, eden_io

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Annotate graphs using a predictive model.
"""

def setup_parameters(parser):
	parser.add_argument("-i", "--input-file",  
    	dest = "input_file",
    	help = "File name with graphs.", 
    	required = True)
	parser.add_argument("-f", "--format",  choices = ["gspan", "node_link_data", "obabel"],
    	help = "File format.", 
    	default = "gspan")
	parser.add_argument("-o", "--output-dir", 
		dest = "output_dir_path", 
		help = "Path to output directory.",
		default = "out")	
	parser.add_argument( "-w", "--reweight-factor",
		dest = "reweight",
		type = int, 
		help = """
            Update the 'weight' information as a linear combination of the previuous weight and 
            the absolute value of the margin. 
            If reweight = 0 then do not update.
            If reweight = 1 then discard previous weight information and use only abs(margin)
            If reweight = 0.5 then update with the aritmetic mean of the previous weight information 
            and the abs(margin)
            """,
        default = 1)
	parser.add_argument("-v", "--verbosity", 
		action = "count",
		help = "Increase output verbosity")


	return parser


def main(args):
	"""
	Annotate graphs using a predictive model.
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
	
	#serialize graphs
	serialized_list = [ line for line in node_link_data.eden_to_node_link_data(ann_g_list)]
	
	#save results
	full_out_file_name = os.path.join(args.output_dir_path, "annotated_node_link_data")
	with open(full_out_file_name, "w") as f:
		f.write("\n".join(serialized_list))


if __name__  == "__main__":
	args = setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Program: %s' % sys.argv[0])
	logging.info('Started')
	main(args)
	logging.info('Finished')