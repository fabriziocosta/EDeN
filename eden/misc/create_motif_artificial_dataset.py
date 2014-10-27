#!/usr/bin/env python

import sys
import os
import argparse
import random

def random_sequence(size = None, list_of_chars = None):
    num_chars = len(list_of_chars)
    seq = []
    #compose a list of repreated occurrences of characters in list_of_chars
    for c in list_of_chars:
        seq += [c]*(size/num_chars)
    #shuffle the list
    random.shuffle(seq)
    return ''.join(seq)


def positive_dataset(seed_seq_list = None, seq_size = None, dataset_size = None, list_of_chars = None):
    dataset = []
    num_seeds = len(seed_seq_list)
    seed_size = len(seed_seq_list[0])
    effective_emi_size = (seq_size - seed_size * num_seeds) / (2 * num_seeds) + 1
    for i in range(dataset_size):
        seq = ''
        for seed_seq in seed_seq_list:
            left_seq = random_sequence(size = effective_emi_size, list_of_chars = list_of_chars)
            right_seq = random_sequence(size = effective_emi_size, list_of_chars = list_of_chars)
            single_seq = left_seq + seed_seq + right_seq
            seq += single_seq
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
        default = 7)
    parser.add_argument( "-n", "--num-motifs",
        dest = "num_motifs",
        type = int,
        help = "Number of motifs per sequence.",
        default = 1)
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
    seed_seq_list = []
    for i in range(args.num_motifs):
        seed_seq = random_sequence(size = seed_size, list_of_chars = "AUCG")
        seed_seq_list += [seed_seq]
        print 'Motif %d: %s'% (i,seed_seq)
    dataset_pos = positive_dataset(seed_seq_list = seed_seq_list, seq_size = args.sequence_size, dataset_size = args.dataset_size, list_of_chars = "AUCG")
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
