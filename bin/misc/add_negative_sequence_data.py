#!/usr/bin/env python

import sys
import os
import argparse
import random

if __name__  == "__main__":
    parser = argparse.ArgumentParser(description = "Add randomly shuffled data and output an additional target file", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-file",
        dest = "input_file",
        help = "File name with sequences.", 
        required = True)
    parser.add_argument( "-n", "--num-negative-sequences-per-positive-sequence",
        dest = "num_negative_sequences_per_positive_sequence",
        type = int,
        help = "Number of sequences to generate for each positive sequence.",
        default = 1)
    args = parser.parse_args()

    dataset_filename = "dataset.seq"
    d = open(dataset_filename, "w")
    target_filename = "dataset.target"
    t = open(target_filename, "w")
    with open(args.input_file,'r') as f:
        for line in f:
            d.write(line.strip()+'\n')
            t.write('1\n')
            seq = [c for c in line.strip()]
            for i in range(args.num_negative_sequences_per_positive_sequence):
                random.shuffle(seq)
                seq_str = ''.join(seq)
                d.write(seq_str+'\n')
                t.write('-1\n')
    d.close()
    t.close()
    print("Written files %s and %s" % (dataset_filename, target_filename))
