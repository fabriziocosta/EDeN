#!/usr/bin/env python

import sys
import os
import argparse
import random

def random_RNA(size):
    RNAseq = ['A']*(size/4) + ['U']*(size/4) + ['C']*(size/4) + ['G']*(size/4)
    random.shuffle(RNAseq) 
    return ''.join(RNAseq)


def positive_dataset(seed_seq='', seq_size = '', dataset_size = ''):
    dataset = []
    seed_size = len(seed_seq)
    for i in range(dataset_size):
        effective_emi_size = (seq_size - seed_size)/2
        left_seq = random_RNA(effective_emi_size)
        right_seq = random_RNA(effective_emi_size)
        seq = left_seq + seed_seq + right_seq
        dataset += [seq]
    return dataset


def negative_dataset(pos_dat):
    dataset = []
    for seq in pos_dat:
        seq_list = [c for c in seq]
        random.shuffle(seq_list)
        dataset += [''.join(seq_list)]
    return dataset   


if __name__  == "__main__":
	parser = argparse.ArgumentParser(description = "Create artificial dataset for sequential motif finding", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument( "-k", "--min-motif-size",
		dest = "min_motif_size",
		type = int,
		help = "Minimal length of the motif.",
        default = 5)
	parser.add_argument( "-s", "--sequence-size",
		dest = "sequence_size",
		type = int,
		help = "Length of the sequences.",
        default = 30)
	parser.add_argument( "-d", "--dataset-size",
		dest = "dataset_size",
		type = int,
		help = "Number of sequences containing the motif.",
        default = 100)
	args = parser.parse_args()

	seed_size=args.min_motif_size
	seed_seq = random_RNA(seed_size)
	print 'Motif: %s'% seed_seq
	dataset_pos = positive_dataset(seed_seq = seed_seq, seq_size = args.sequence_size, dataset_size = args.dataset_size)
	dataset_neg = negative_dataset(dataset_pos)
	target = ["1"]*len(dataset_pos) + ["-1"]*len(dataset_neg)
	dataset = dataset_pos + dataset_neg
	
	dataset_filename = "dataset.seq"
	with open(dataset_filename, "w") as f:
		f.write('\n'.join(dataset))
	print "Dataset written to file: %s" % dataset_filename

	target_filename = "dataset.target"
	with open(target_filename, "w") as f:
		f.write('\n'.join(target))
	print "Target written to file: %s" % target_filename

	dataset_pos_filename = "dataset_positive.seq"
	with open(dataset_pos_filename, "w") as f:
		f.write('\n'.join(dataset_pos))
	print "Dataset written to file: %s" % dataset_pos_filename