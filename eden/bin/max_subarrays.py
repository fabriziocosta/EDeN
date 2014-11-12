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
Extract maximal subarrays by annotating graphs using a predictive model.

Note: the program assumes the existence of an output directory specified via --output-dir
that contains a file called 'model' and a file called 'vectorizer'. You can create these files by 
first running the fit.py program.  
"""

def setup_parameters(parser):
	parser.add_argument("-i", "--input-file",
		dest = "input_file",
    	help = "File name with graphs.", 
    	required = True)
	parser.add_argument( "-k", "--min-subarray-size",
		dest = "min_subarray_size",
		type = int,
		help = "Minimal length of the subarray to extract.",
        default = 7)
	parser.add_argument( "-m", "--max-subarray-size",
		dest = "max_subarray_size",
		type = int,
		help = "Maximal length of the subarray to extract. Use -1 for no limit.",
        default = -1)
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
	parser.add_argument("-v", "--verbosity", 
		action = "count",
		help = "Increase output verbosity")
	return parser


def main(args):
	"""
	Extract maximal subarrays by annotating graphs using a predictive model.
	"""
	#load vectorizer
	vec = eden_io.load(output_dir_path = args.output_dir_path, out_file_name = "vectorizer")
	logging.info('Vectorizer: %s' % vec)

	#load predictive model
	clf = eden_io.load(output_dir_path = args.output_dir_path, out_file_name = "model")
	logging.info('Model: %s' % clf)

	#initialize annotator
	ann = graph.Annotator(estimator = clf, vectorizer = vec, reweight = args.reweight)
	
	#load data
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)
	
	#annotate
	ann_g_list = [g for g in  ann.transform(g_it)]
	
	#extract_subarrays for each graph
	subarray_list = []
	for g in ann_g_list:
		subarrays = iterated_maximum_subarray.compute_max_subarrays(graph = g, min_subarray_size = args.min_subarray_size, max_subarray_size = args.max_subarray_size)
		if subarrays:
			subarray_list += subarrays

	#save results
	full_out_file_name = os.path.join(args.output_dir_path, "subarrays.data")
	with open(full_out_file_name, "w") as f:
		for subarray_item in subarray_list:
			subarray_str = "".join(subarray_item['subarray'])
			if subarray_str:
				seq_str = "".join(subarray_item['seq'])
				line = "subarray:%s score:%0.4f begin:%d end:%d size:%d seq:%s\n" % (subarray_str,subarray_item['score'],subarray_item['begin'],subarray_item['end'],subarray_item['size'],seq_str)
				f.write(line)
	full_out_file_name = os.path.join(args.output_dir_path, "subarrays")
	with open(full_out_file_name, "w") as f:
		for subarray_item in subarray_list:
			subarray_str = "".join(subarray_item['subarray'])
			if subarray_str:
				line = "%s\n" % subarray_str
				f.write(line)


if __name__  == "__main__":
	args = setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Program: %s' % sys.argv[0])
	logging.info('Started')
	logging.info('Parameters: %s' % args)
	main(args)
	logging.info('Finished')