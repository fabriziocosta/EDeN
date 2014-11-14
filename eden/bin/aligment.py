#!/usr/bin/env python

import sys
import os
import logging
import math
import pylab as pl

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
import numpy as np
from scipy.sparse import csr_matrix

from eden import graph
from eden.converters import dispatcher
from eden.util import argument_parser, setup, eden_io

from eden import NeedlemanWunsh

DESCRIPTION = """
Explicit Decomposition with Neighborhood Utility program.
Compute optimal alignment using Needleman-Wunsh algorithm on annotated graphs.

Note: the program assumes the existence of an output directory specified via --output-dir
that contains a file called 'model' and a file called 'vectorizer'. You can create these files by 
first running the fit.py program.  
"""

def setup_parameters(parser):
	parser.add_argument("-i", "--input-file",
		dest = "input_file",
    	help = "File name with graphs.", 
    	required = True)
	parser.add_argument( "-g", "--gap-penalty",
		dest = "gap_penalty",
		type = float,
		help = "Cost of inserting a gap in the alignment.",
        default = -1)
	parser.add_argument( "-r", "--importance-threshold",
		dest = "importance_threshold",
		type = float,
		help = "Importance threshold to mark significant parts of the alignment.",
        default = 1)
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
	parser.add_argument("-x", "--output-dendrogram", 
		dest = "output_dendrogram",
		help = "Output a pdf and svg dendrogram using the alignment score.",
		action = "store_true")	
	parser.add_argument("--plot-size",
		dest = "plot_size",
		type = int, 
		default = 120,
		help = "Size of the plot area of the dendrogam picture.")
	parser.add_argument("-l", "--linkage",  choices = [ "median","centroid","weighted","single","ward","complete","average"],
    	help = """Which linkage criterion to use for the dendrogram. 
    	The linkage criterion determines which distance to use between sets of observation. 
    	The algorithm will merge the pairs of cluster that minimize this criterion.
        - ward minimizes the variance of the clusters being merged.
        - average uses the average of the distances of each observation of the two sets.
        - complete or maximum linkage uses the maximum distances between all observations of the two sets.
        - single also known as the Nearest Point Algorithm.
        - weighted also called WPGMA.
        - centroid when two clusters s and t are combined into a new cluster u, the new centroid 
        is computed over all the original objects in clusters s and t. The distance then becomes 
        the Euclidean distance between the centroid of u and the centroid of a remaining cluster 
        v in the forest. This is also known as the UPGMC algorithm.
        - median when two clusters s and t are combined into a new cluster u, the average of 
        centroids s and t give the new centroid u. This is also known as the WPGMC algorithm.
    	""", 
    	default = "average")
	return parser


def to_matrix(feature_dict, feature_size):
    D = []
    R = []
    C = []
    for i,v in enumerate(feature_dict):
        data = v.values()
        col = [int(k) for k in v.keys()]
        row = [i]*len(col)
        D += data
        R += row
        C += col
    X = csr_matrix( (D,(R,C)), shape = (max(R)+1,feature_size))
    return X


def similarity_matrix(svecA, svecB, nbits):
	#convert dictionary into sparse vector
	XA = to_matrix(svecA, 2 ** nbits + 1)
	XB = to_matrix(svecB, 2 ** nbits + 1)
	#compute dot product
	S = XA.dot(XB.T).todense()
	return S


def exact_match_alignment_score(alnA, alnB):
	score = 0.0
	for a,b in zip(alnA,alnB):
		if a == b:
			score += 1
	return score / len(alnA) 


def insert_gaps_in_importance_vector(alnA, graphA):
	impA = [d['importance'] for u,d in graphA.nodes(data = True)]
	gapped_impA = []
	index = 0
	for c in alnA:
		if c == '-':
			gapped_impA += [0]
		else:
			gapped_impA += [impA[index]]
			index += 1
	return gapped_impA


def soft_match_alignment_score(alnA, alnB, graphA, graphB):
	impA = np.array(insert_gaps_in_importance_vector(alnA, graphA))
	impB = np.array(insert_gaps_in_importance_vector(alnB, graphB))
	return impA.dot(impB)/math.sqrt(impA.dot(impA)*impB.dot(impB))


def align(instA = None, instB = None, gap_penalty = None, nbits = None):
	#NOTE: the sequential order of nodes in the graph are used as the sequential constrain   
	#extract sequence of labels
	seqA = [d['label'] for u,d in instA.nodes(data = True)]
	strA = ''.join(seqA)
	seqB = [d['label'] for u,d in instB.nodes(data = True)]
	strB = ''.join(seqB)
	#extact vector representation
	svecA = [d['vector'] for u,d in instA.nodes(data = True)]
	svecB = [d['vector'] for u,d in instB.nodes(data = True)]
	#compute similarity matrix
	S = similarity_matrix(svecA, svecB, nbits)
	#compute alignment matrix
	F = NeedlemanWunsh.needleman_wunsh(strA,strB,S,gap_penalty)
	#compute traceback
	alnA,alnB = NeedlemanWunsh.trace_back(strA,strB,S,gap_penalty,F)
	return (alnA,alnB)


def insert_gaps(seq, gapped_seq):
	out_seq = ''
	i = 0
	for c in gapped_seq:
		if c == '-':
			out_seq += ' '
		else:
			out_seq += seq[i]
			i = i + 1
	return out_seq


def mark_exact_match(seqA,seqB):
	out_seq = ''
	for a,b in zip(seqA,seqB):
		if a == b:
			out_seq += '|'
		else:
			out_seq += ' '
	return out_seq 


def len_seq(seq):
	l = 0
	for c in seq:
		if c != '-':
			l += 1
	return l


def extract_seq(seq):
	out_seq = ''
	for c in seq:
		if c != '-':
			out_seq += c
	return out_seq


def extract_importance_string(graph, threshold):
	out_str = ''
	for u,d in graph.nodes(data = True):
		imp = d['importance']
		if imp > 2 * threshold:
			out_str += '*'
		elif imp > 1.5 * threshold:
			out_str += '+'
		elif imp > threshold:
			out_str += '-'
		else:
			out_str += ' '
	return out_str


def alignment_score_matrix(alignment_list, size = None):
	F = np.zeros((size,size))
	for aln in alignment_list:
		(exact_match_score,soft_match_score,i,alnA,importanceA,j,alnB,importanceB) = aln
		F[i,j] = soft_match_score
	#make symmetric
	F = F + F.T
	return F


def output(args, alignment_list):
	full_out_file_name = os.path.join(args.output_dir_path, "alignments")
	with open(full_out_file_name, "w") as f:
		for aln in alignment_list:
			(exact_match_score,soft_match_score,i,alnA,importanceA,j,alnB,importanceB) = aln
			header_str = "ID:%s vs ID:%s exact_match_score: %.4f soft_match_score: %.4f \n"%(i,j,exact_match_score,soft_match_score)
			seqA_str = "ID:%s len:%d %s\n" % (i,len_seq(alnA),extract_seq(alnA))
			seqB_str = "ID:%s len:%d %s\n" % (j,len_seq(alnB),extract_seq(alnB))
			alnA_str = "%s\n" % alnA
			alnB_str = "%s\n" % alnB
			importanceA_str = "%s\n" % ''.join(importanceA)
			importanceB_str = "%s\n" % ''.join(importanceB)
			exact_match_str = "%s\n" % mark_exact_match(alnA,alnB)
			f.write(header_str)
			f.write(seqA_str)
			f.write(seqB_str)
			f.write("\n")
			f.write(importanceA_str)
			f.write(alnA_str)
			f.write(exact_match_str)
			f.write(alnB_str)
			f.write(importanceB_str)
			f.write("\n\n")


def output_dendrogram(args, alignment_score_matrix):
	#compute distance matrix
 	D = 1 - alignment_score_matrix
	Z = linkage(D, method = args.linkage)
	
	#compute dendrogram
	fig = pl.figure(figsize = (args.plot_size//10,args.plot_size))
	dendrogram(Z, orientation = 'right')

	#save plot
	out_file_name = 'dendrogram'
	if not os.path.exists(args.output_dir_path) :
		os.mkdir(args.output_dir_path)
	full_out_file_name = os.path.join(args.output_dir_path, out_file_name)
	fig.savefig(full_out_file_name + ".pdf")
	fig.savefig(full_out_file_name + ".svg")
	logging.info('Saved file: %s' % out_file_name)


def main(args):
	"""
	Compute optimal alignment using Needleman-Wunsh algorithm on annotated graphs.
	"""
	#load vectorizer
	vec = eden_io.load(output_dir_path = args.output_dir_path, out_file_name = "vectorizer")
	logging.info('Vectorizer: %s' % vec)
	nbits = vec.nbits
	logging.info('nbits: %s' % nbits)

	#load predictive model
	clf = eden_io.load(output_dir_path = args.output_dir_path, out_file_name = "model")
	logging.info('Model: %s' % clf)

	#initialize annotator
	ann = graph.Annotator(estimator = clf, vectorizer = vec, reweight = args.reweight, annotate_vertex_with_vector = True)
	
	#load data
	g_it = dispatcher.any_format_to_eden(input_file = args.input_file, format = args.format)
	
	#annotate
	ann_g_list = [g for g in  ann.transform(g_it)]
	
	#compute alignment for each graph pair
	alignment_list = []
	n = len(ann_g_list)
	for i in range(n):
		importanceA = extract_importance_string(ann_g_list[i], args.importance_threshold)
		for j in range(i+1,n):
			importanceB = extract_importance_string(ann_g_list[j], args.importance_threshold)
			(alnA,alnB) = align(instA = ann_g_list[i], instB = ann_g_list[j], gap_penalty = args.gap_penalty, nbits = nbits) 
			impA_str = insert_gaps(importanceA, alnA)
			impB_str = insert_gaps(importanceB, alnB)
			exact_match_score = exact_match_alignment_score(alnA,alnB)
			soft_match_score = soft_match_alignment_score(alnA, alnB, ann_g_list[i], ann_g_list[j])
			alignment_list += [(exact_match_score,soft_match_score,i,alnA,impA_str,j,alnB,impB_str)]

	#save results
	output(args, alignment_list)	

	#make alignemnt score matrix
	F = alignment_score_matrix(alignment_list, size = len(ann_g_list))
	#save matrix
	eden_io.store_matrix(matrix = F, output_dir_path = args.output_dir_path, out_file_name = "alignment_score_matrix", output_format = "MatrixMarket")

	if args.output_dendrogram:
		#dendrogram
		output_dendrogram(args, F)


if __name__  == "__main__":
	args = setup.setup(DESCRIPTION, setup_parameters)
	logging.info('Program: %s' % sys.argv[0])
	logging.info('Started')
	logging.info('Parameters: %s' % args)
	main(args)
	logging.info('Finished')