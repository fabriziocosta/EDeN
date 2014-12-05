#!/usr/bin/env python

import sys
import os
import time

from eden.util import setup
from eden.graphicalizer.graph import node_link_data


DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Format converter.
"""

def instance_to_eden(input = None, input_type = None, tool = None, options = dict()):
	#TODO: scan folders and subfolders in graphicalizer and find tool that has the same name
	if tool == "gspan":
		from eden.graphicalizer.graph import gspan
		g_it = gspan.gspan_to_eden(input = input, input_type = input_type, options = options)
	elif tool == "node_link_data":
		from eden.graphicalizer.graph import node_link_data
		g_it = node_link_data.node_link_data_to_eden(input = input, input_type = input_type, options = options)
	elif tool == "sequence":
		from eden.graphicalizer.graph import sequence
		g_it = sequence.sequence_to_eden(input = input, input_type = input_type, options = options)
	elif tool == "obabel":
		from eden.graphicalizer.molecule import obabel
		g_it = obabel.obabel_to_eden(input = input, input_type = input_type, options = options)
	elif tool == "FASTA":
		from eden.graphicalizer.RNA import FASTA
		g_it = FASTA.FASTA_to_eden(input = input, input_type = input_type, options = options)
	else:
		raise Exception('Unknown tool: %s' % tool)
	return g_it


def setup_parameters(parser):
    parser.add_argument("-i", "--input-file",
    	dest = "input_file",
		help = "Path to your graph file.", 
		required = True)
    parser.add_argument("-t", "--input-type",  choices = ['url','file'],
		dest = "input_type",
		help = "If type is 'url' then 'input' is interpreted as a URL pointing to a file. If type is 'file' then 'input' is interpreted as a file name.",
		default = "file")
    parser.add_argument("-f", "--graphicalizer-tool",  choices = ["gspan", "node_link_data", "obabel", "sequence", "FASTA"],
		dest = "graphicalizer_tool",
		help = "Tool name for the graphicalization phase, i.e. the transformation of instances from the original data format into graphs.", 
		default = "gspan")
    parser.add_argument("-o", "--output-dir", 
		dest = "output_dir_path", 
		help = "Path to output directory.",
		default = "out")
    parser.add_argument("-v", "--verbosity", 
		action = "count",
		help = "Increase output verbosity")
    return parser


def convert(args):
	#load data
	g_it = instance_to_eden(input = args.input_file, input_type = args.input_type, tool = args.graphicalizer_tool)
	#write data
	out_file_name = 'data.nld'
	if not os.path.exists(args.output_dir_path) :
		os.mkdir(args.output_dir_path)
	full_out_file_name = os.path.join(args.output_dir_path, out_file_name)
	with open(full_out_file_name, "w") as f:
		for line in node_link_data.eden_to_node_link_data(g_it):
			f.write("%s\n"%line)
	logger.info("Written file: %s/%s",args.output_dir_path, out_file_name)



if __name__  == "__main__":
	start_time = time.clock()
	args = setup.arguments_parser(DESCRIPTION, setup_parameters)
	logger = setup.logger(logger_name = "convert", filename = "log", verbosity = args.verbosity)

	logger.info('-'*80)
	logger.info('Program: %s' % sys.argv[0])
	logger.info('Parameters: %s' % args.__dict__)
	try:
		convert(args)
	except Exception:
		import datetime
		curr_time = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
		logger.exception("Program run failed on %s" % curr_time)
	finally:
		end_time = time.clock()
		logger.info('Elapsed time: %.1f sec',end_time - start_time)	