#!/usr/bin/env python

import subprocess as sp
from itertools import tee
import numpy as np
import random

import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels

from eden import AbstractVectorizer
from eden.path import Vectorizer as SeqVectorizer
from eden.graph import Vectorizer as GraphVectorizer
from eden.converter.rna import sequence_dotbracket_to_graph
from eden.converter.fasta import seq_to_networkx

import logging
logger = logging.getLogger(__name__)


def convert_seq_to_fasta_str(seq):
    return '>%s\n%s\n' % seq


def extract_aligned_seed(header, out):
    text = out.strip().split('\n')
    seed = ''
    for line in text:
        if header in line:
            seed += line.strip().split()[1]
    return seed


def extract_struct_energy(out):
    text = out.strip().split('\n')
    struct = text[1].strip().split()[0]
    energy = text[1].strip().split()[1:]
    energy = ' '.join(energy).replace('(', '').replace(')', '')
    energy = energy.split('=')[0]
    energy = float(energy)
    return struct, energy


def make_seq_struct(seq, struct):
    clean_seq = ''
    clean_struct = ''
    for seq_char, struct_char in zip(seq, struct):
        if seq_char == '-' and struct_char == '.':
            pass
        else:
            clean_seq += seq_char
            clean_struct += struct_char
    return clean_seq, clean_struct


class Vectorizer(AbstractVectorizer):

    def __init__(self,
                 sequence_vectorizer_complexity=3,
                 graph_vectorizer_complexity=2,
                 n_neighbors=5,
                 sampling_prob=.5,
                 n_iter=5):
        self.sequence_vectorizer = SeqVectorizer(complexity=sequence_vectorizer_complexity,
                                                 normalization=False,
                                                 inner_normalization=False)
        self.graph_vectorizer = GraphVectorizer(complexity=graph_vectorizer_complexity)
        self.n_neighbors = n_neighbors
        self.sampling_prob = sampling_prob
        self.n_iter = n_iter
        self.nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors)

    def fit(self, seqs):
        # store seqs
        self.seqs = list(seqs)
        data_matrix = self.sequence_vectorizer.transform(self.seqs)
        # fit nearest_neighbors model
        self.nearest_neighbors.fit(data_matrix)
        return self

    def fit_transform(self, seqs):
        seqs, seqs_ = tee(seqs)
        return self.fit(seqs_).transform(seqs)

    def transform(self, seqs):
        seqs = list(seqs)
        graphs_ = self.graphs(seqs)
        data_matrix = self.graph_vectorizer.transform(graphs_)
        return data_matrix

    def graphs(self, seqs):
        for seq, neighs in self._compute_neighbors(seqs):
            if self.n_iter > 1:
                header, sequence, struct, energy = self._optimize_struct(seq, neighs)
            else:
                header, sequence, struct, energy = self._align_sequence_structure(seq, neighs)
            graph = self._seq_to_eden(header, sequence, struct, energy)
            yield graph

    def _optimize_struct(self, seq, neighs):
        structs = []
        results = []
        for i in range(self.n_iter):
            new_neighs = self._sample_neighbors(neighs)
            header, sequence, struct, energy = self._align_sequence_structure(seq, new_neighs)
            results.append((header, sequence, struct, energy))
            structs.append(struct)
        instance_id = self._most_representative(structs)
        selected = results[instance_id]
        return selected

    def _most_representative(self, structs):
        # compute kernel matrix with sequence_vectorizer
        data_matrix = self.sequence_vectorizer.transform(structs)
        kernel_matrix = pairwise_kernels(data_matrix, metric='rbf', gamma=1)
        # compute instance density as 1 over average pairwise distance
        density = np.sum(kernel_matrix, 0) / data_matrix.shape[0]
        # compute list of nearest neighbors
        max_id = np.argsort(-density)[0]
        return max_id

    def _sample_neighbors(self, neighs):
        out_neighs = []
        # insert one element at random
        out_neighs.append(random.choice(neighs))
        # add other elements sampling without replacement
        for neigh in neighs:
            if random.random() < self.sampling_prob:
                out_neighs.append(neigh)
        return out_neighs

    def _align_sequence_structure(self, seq, neighs):
        header = seq[0]
        str_out = convert_seq_to_fasta_str(seq)
        for neigh in neighs:
            str_out += convert_seq_to_fasta_str(neigh)
        cmd = 'echo "%s" | muscle -clwstrict -quiet' % (str_out)
        out = sp.check_output(cmd, shell=True)
        seed = extract_aligned_seed(header, out)
        cmd = 'echo "%s" | RNAalifold -noPS' % (out)
        out = sp.check_output(cmd, shell=True)
        struct, energy = extract_struct_energy(out)
        clean_seq, clean_struct = make_seq_struct(seed, struct)
        return header, clean_seq, clean_struct, energy

    def _compute_neighbors(self, seqs):
        seqs = list(seqs)
        data_matrix = self.sequence_vectorizer.transform(seqs)
        # find neighbors
        distances, neighbors = self.nearest_neighbors.kneighbors(data_matrix)
        # for each seq
        for seq, neighs in zip(seqs, neighbors):
            neighbor_seqs = [self.seqs[neigh] for neigh in neighs]
            yield seq, neighbor_seqs

    def _seq_to_eden(self, header, sequence, struct, energy):
        graph = nx.Graph()
        graph.graph['id'] = header
        graph.graph['info'] = 'muscle+RNAalifold energy=%.3f' % (energy)
        graph.graph['energy'] = energy
        graph.graph['sequence'] = sequence
        graph = sequence_dotbracket_to_graph(seq_info=sequence, seq_struct=struct)
        if graph.number_of_nodes() < 2:
            graph = seq_to_networkx(header, sequence)
        return graph
