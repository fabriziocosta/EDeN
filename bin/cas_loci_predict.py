#!/usr/bin/env python

import networkx as nx
from itertools import tee
import os
from time import time
import numpy as np

from sklearn.linear_model import SGDClassifier
import joblib

import argparse
import logging
import logging.handlers

from eden.util import configure_logging
from eden.util import serialize_dict
from eden.graph import Vectorizer
from eden.util.iterated_maximum_subarray import compute_iterated_maximum_subarray
from eden.util import vectorize


logger = logging.getLogger(__name__)

description = """
CasLociPredictor


Example usage:
# for fitting a predictive model:
./cas_loci_predict.py -x -v fit -c 2 -b 13 -n 30 -s 3 -l 'clust_id' 'domain_scores' \
-i ../data/Traindataset_Cas_loci_TypeI_system_only.Feautres.tab \
-m mod_type_I.md -g ../data/ProteinID_DomainID_Bitscore_TypeI.tab

# for predicting using a fit model:
./cas_loci_predict.py -x predict -i ../data/Testdataset_Cas_loci_TypeI_system_only.Feautres.tab \
-g ../data/ProteinID_DomainID_Bitscore_TypeI.tab -m out/mod_type_I.md
cat out/summary.txt
"""

epilog = """
Author: Fabrizio Costa
Copyright: 2015
License: GPL
Maintainer: Fabrizio Costa
Email: costa@informatik.uni-freiburg.de
Status: Production

Cite: ...
"""


def file_to_domain(fname):
    gene_domain_scores = dict()
    with open(fname) as f:
        for counter, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            gene_id = tokens[0]
            domain_score_dict = dict()
            domain_scores = tokens[1].split(',')
            for domain_score in domain_scores:
                domain_score_tokens = domain_score.split(':')
                domain_id = domain_score_tokens[0]
                bit_score = float(domain_score_tokens[1])
                domain_score_dict[domain_id] = bit_score
            gene_domain_scores[gene_id] = domain_score_dict
    return gene_domain_scores


def file_to_loci(fname):
    old_group_id = None
    block_locus = []
    with open(fname) as f:
        for counter, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if counter == 0:
                # process column names
                column_names = line.split()
            if counter == 1:
                # process types
                column_types = line.split()
            if counter > 1:
                group_id = line.split('\t')[1]
                if old_group_id is None or old_group_id == group_id:
                    block_locus.append(line)
                else:  # starts new cas loci
                    yield column_names, column_types, block_locus
                    block_locus = []
                    block_locus.append(line)
                old_group_id = group_id
    if block_locus:
        yield column_names, column_types, block_locus


def block_to_block_structure(column_names, column_types, block):
    block_structure = []
    for line in block:
        block_data = dict()
        tokens = line.split('\t')
        for column_name, column_type, token in zip(column_names, column_types, tokens):
            if column_type == 'str':
                value = str(token)
            elif column_type == 'int':
                value = [float(token)]
            elif column_type == 'float':
                value = [float(token)]
            else:
                raise Exception('Non supported type: % s' % column_type)
            block_data[column_name] = value
        block_structure.append(block_data)
    return block_structure


def update_block_with_gene_domain_scores(block_structure,
                                         gene_domain_scores,
                                         gene_id_key='gene_id',
                                         domain_scores_key='domain_scores'):
    new_block_structure = []
    for tokens in block_structure:
        domain_scores = gene_domain_scores[tokens[gene_id_key]]
        tokens[domain_scores_key] = domain_scores
        new_block_structure.append(tokens)
    return new_block_structure


def block_structure_to_eden(block_structure,
                            column_names=None,
                            target_key='target',
                            id_key='id',
                            position_key='order_to_signature'):
    graph = nx.Graph()
    graph.graph['id'] = block_structure[0][id_key][0]
    id_counter = 0
    backbone_counter = -1
    prev_backbone_id = None
    for block_data in block_structure:
        for i, column_name in enumerate(column_names):
            if i == 0:
                backbone_id = id_counter
                backbone_counter += 1
            graph.add_node(id_counter,
                           position=block_data[position_key][0],
                           label=block_data[column_name],
                           entity=column_name,
                           target=block_data[target_key],
                           backbone_counter=backbone_counter)
            if i != 0:
                graph.add_edge(id_counter - 1, id_counter, label='-')
            id_counter += 1
        if prev_backbone_id is not None:
            graph.add_edge(prev_backbone_id, backbone_id, label=':')
        prev_backbone_id = backbone_id
    return graph


def fragment_graph(graph, window_size=1):
    id = graph.graph['id']
    # extract backbone size
    backbone_size = max([graph.node[u]['backbone_counter'] for u in graph])

    # for loop on windows
    for i in range(backbone_size + 1):
        left = i - window_size
        right = i + window_size
        if left >= 0 and right <= backbone_size:
            window_ids = range(left, right + 1)
            node_ids = [u for u in graph if graph.node[u]['backbone_counter'] in window_ids]
            fragment = graph.subgraph(node_ids)
            positions = [fragment.node[u]['position'] for u in fragment]
            min_position = min(positions)
            max_position = max(positions)
            center_position = (max_position - min_position) / 2 + min_position
            fragment.graph['position'] = center_position
            fragment.graph['id'] = id
            yield fragment


def construct_loci_graphs(fname,
                          selected_column_names=None,
                          gene_domain_score_fname=None):
    gene_domain_scores = file_to_domain(gene_domain_score_fname)
    logger.debug('read file: %s' % gene_domain_score_fname)
    for column_names, column_types, block in file_to_loci(fname):
        block_structure = block_to_block_structure(column_names, column_types, block)
        if gene_domain_score_fname is not None:
            block_structure = update_block_with_gene_domain_scores(block_structure, gene_domain_scores)
        graph = block_structure_to_eden(block_structure, column_names=selected_column_names)
        yield graph


def extract_target(graphs):
    for graph in graphs:
        graph_targets = [graph.node[n]['target'][0] for n in graph if 'target' in graph.node[n]]
        target_average = np.mean(graph_targets)
        if target_average >= 0.5:
            target = 1
        else:
            target = -1
        yield target


def concatenate_loci_graphs(graph_lists):
    for graphs in graph_lists:
        for graph in graphs:
            yield graph


def save_output(text=None, output_dir_path=None, out_file_name=None):
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    full_out_file_name = os.path.join(output_dir_path, out_file_name)
    with open(full_out_file_name, 'w') as f:
        for line in text:
            f.write("%s\n" % str(line).strip())
    logger.info("Written file: %s (%d lines)" %
                (full_out_file_name, len(text)))


class CasLociPredictor(object):

    def __init__(self,
                 selected_column_names=None,
                 window_size=2,
                 complexity=4,
                 nbits=10,
                 n=100,
                 label_size=5,
                 min_subarray_size=3,
                 max_subarray_size=5):
        self.selected_column_names = selected_column_names
        self.window_size = window_size
        self.complexity = complexity
        self.nbits = nbits
        self.n = n
        self.label_size = label_size
        self.min_subarray_size = min_subarray_size
        self.max_subarray_size = max_subarray_size
        self.vectorizer = None
        self.estimator = None

    def save(self, model_name):
        joblib.dump(self, model_name, compress=1)

    def load(self, model_name):
        self.__dict__.update(joblib.load(model_name).__dict__)

    def fit(self, input_file=None, gene_domain_score_file=None):
        loci_graphs = construct_loci_graphs(input_file,
                                            selected_column_names=self.selected_column_names,
                                            gene_domain_score_fname=gene_domain_score_file)
        loci_graphs = list(loci_graphs)
        logger.debug('processing %d graphs' % len(loci_graphs))
        graph_lists = {fragment_graph(loci_graph, window_size=self.window_size)
                       for loci_graph in loci_graphs}
        graphs = concatenate_loci_graphs(graph_lists)
        graphs, graphs_ = tee(graphs)
        train_y = np.array(list(extract_target(graphs_)))
        self.vectorizer = Vectorizer(complexity=self.complexity,
                                     n=self.n,
                                     label_size=self.label_size,
                                     nbits=self.nbits)
        self.vectorizer.fit(loci_graphs)
        train_data_matrix = vectorize(graphs,
                                      vectorizer=self.vectorizer,
                                      fit_flag=False,
                                      n_blocks=5, block_size=None, n_jobs=8)
        logger.debug('train data matrix #instances: %d  #features: %d' %
                     (train_data_matrix.shape[0], train_data_matrix.shape[1]))
        # induce a predictive model
        self.estimator = SGDClassifier(average=True, class_weight='balanced', shuffle=True, n_jobs=-1)
        self.estimator.fit(train_data_matrix, train_y)

    def predict(self, input_file=None, gene_domain_score_file=None):
        graphs = construct_loci_graphs(input_file,
                                       selected_column_names=self.selected_column_names,
                                       gene_domain_score_fname=gene_domain_score_file)
        graph_lists = [fragment_graph(graph, window_size=self.window_size) for graph in graphs]
        scores = self.evaluate(graph_lists)
        return scores

    def loci_region(self, score=None):
        seq = [str(x) for x in range(len(score))]
        subarrays = compute_iterated_maximum_subarray(seq=seq,
                                                      score=score,
                                                      min_subarray_size=self.min_subarray_size,
                                                      max_subarray_size=self.max_subarray_size,
                                                      margin=0,
                                                      output='full')
        subarrays = list(subarrays)
        # NOTE: we return only the first subarray
        # TODO: return multiple subarrays
        if subarrays:
            # NOTE: subarrays[0]['end']-1 because end is one past the last
            return int(subarrays[0]['begin']), int(subarrays[0]['end'] - 1)
        else:
            return None, None

    def overlap_score(self, margins, y):
        # TODO: make the count of the fraction of 1s identified, to cover the case with holes
        bp, ep = self.loci_region(list(margins))
        bt, et = self.loci_region(list(y))
        if bp and ep and bt and et:
            bi = max(bp, bt)
            ei = min(ep, et)
            if (ep - bp) + (et - bt) == 0:
                score = 0
            else:
                score = 2 * (ei - bi) / float(((ep - bp) + (et - bt)))
        else:
            score = 0
        return score, bp, ep

    def get_position(self, graphs):
        centers = []
        for fragment in graphs:
            positions = [fragment.node[u]['position'] for u in fragment]
            min_position = min(positions)
            max_position = max(positions)
            center_position = (max_position - min_position) / 2 + min_position
            centers.append(center_position)
        return centers

    def evaluate(self, graph_lists):
        for i, graphs in enumerate(graph_lists):
            graphs, graphs_ = tee(graphs)
            test_targets = extract_target(graphs_)
            graphs, graphs_ = tee(graphs)
            test_y = np.array(list(test_targets))
            test_data_matrix = self.vectorizer.transform(graphs_)
            graphs, graphs_ = tee(graphs)
            margins = self.estimator.decision_function(test_data_matrix)
            score, begin, end = self.overlap_score(margins, test_y)
            centers = self.get_position(graphs_)
            graph = graphs.next()
            id = graph.graph['id']
            if begin is not None and end is not None:
                yield score, centers[begin], centers[end], id, margins
            else:
                yield score, None, None, id, margins


def main_fit(args):
    pred = CasLociPredictor(selected_column_names=args.selected_column_names,
                            window_size=args.window_size,
                            complexity=args.complexity,
                            n=args.n,
                            label_size=args.label_size,
                            nbits=args.nbits,
                            min_subarray_size=args.min_subarray_size,
                            max_subarray_size=args.max_subarray_size)
    pred.fit(input_file=args.input_file, gene_domain_score_file=args.gene_domain_score_fname)
    if not os.path.exists(args.output_dir_path):
        os.mkdir(args.output_dir_path)
    full_out_file_name = os.path.join(args.output_dir_path, args.model_file)
    pred.save(full_out_file_name)
    logger.debug('Saved model in file: %s' % full_out_file_name)


def main_predict(args):
    pred = CasLociPredictor()
    pred.load(args.model_file)
    scores = pred.predict(input_file=args.input_file, gene_domain_score_file=args.gene_domain_score_fname)
    scores, scores_ = tee(scores)
    score_list = []
    text = []
    text.append('#id score start end')
    for score, begin, end, id, margins in scores:
        score_list.append(score)
        if begin is not None and end is not None:
            line = '%d %.2f %+d %+d' % (id, score, begin, end)
        else:
            line = '%d %.2f  -  -' % (id, score)
        logger.info(line)
        text.append(line)
    save_output(text=text, output_dir_path=args.output_dir_path, out_file_name='predictions.txt')

    text = []
    text.append('#id margins')
    for score, begin, end, id, margins in scores_:
        margins_str = ' '.join(['%.2f' % (val) for i, val in enumerate(margins)])
        line = '%d %s' % (id, margins_str)
        text.append(line)
    save_output(text=text, output_dir_path=args.output_dir_path, out_file_name='margins.txt')

    text = []
    text.append('# instances: %d' % (len(score_list)))
    n_non_scored = sum(1 for s in score_list if s == 0)
    text.append('# non scored: %d' % n_non_scored)
    n_perfect_score = sum(1 for s in score_list if s == 1)
    text.append('# perfect score: %d' % n_perfect_score)
    text.append('avg score: %.2f +- %.2f' % (np.mean(score_list), np.std(score_list)))
    nz_score_list = [s for s in score_list if s > 0]
    text.append('avg non zero score: %.2f +- %.2f' % (np.mean(nz_score_list), np.std(nz_score_list)))
    logger.info('\n'.join(text))
    save_output(text=text, output_dir_path=args.output_dir_path, out_file_name='summary.txt')


def main_dipatch(args):
    if args.which == 'fit':
        main_fit(args)
    elif args.which == 'predict':
        main_predict(args)
    else:
        raise Exception('Unknown mode: %s' % args.which)


def argparse_setup(description, epilog):
    # TODO: write help for commands
    class DefaultsRawDescriptionHelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                                              argparse.RawDescriptionHelpFormatter):
        # To join the behaviour of RawDescriptionHelpFormatter with that of ArgumentDefaultsHelpFormatter
        pass

    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog,
                                     formatter_class=DefaultsRawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbosity",
                        action="count",
                        help="Increase output verbosity")
    parser.add_argument("-x", "--no-logging",
                        dest="no_logging",
                        help="If set, do not log on file.",
                        action="store_true")

    subparsers = parser.add_subparsers(help='commands')
    # base parser
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("-i", "--input-file",
                             dest="input_file",
                             help="Path to file containing input.",
                             required=True)
    base_parser.add_argument("-m", "--model-file",
                             dest="model_file",
                             help="Path to a fit model file.",
                             default="model")
    # TODO: make generic reference file expansion system
    base_parser.add_argument("-g", "--gene-domain-score-file-name",
                             dest="gene_domain_score_fname",
                             help="...",
                             default="gene_domain_score")
    base_parser.add_argument("-o", "--output-dir",
                             dest="output_dir_path",
                             help="Path to output directory.",
                             default="out")
    # fit commands
    fit_parser = subparsers.add_parser('fit', help='Fit commands',
                                       parents=[base_parser],
                                       formatter_class=DefaultsRawDescriptionHelpFormatter)
    fit_parser.set_defaults(which='fit')
    fit_parser.add_argument('-l', '--selected-column-names',
                            dest='selected_column_names',
                            nargs='+',
                            help='....',
                            required=True)
    fit_parser.add_argument("-w", "--window-size",
                            dest="window_size",
                            type=int,
                            help="...",
                            default=2)
    fit_parser.add_argument("-c", "--complexity",
                            dest="complexity",
                            type=int,
                            help="...",
                            default=4)
    fit_parser.add_argument("-b", "--nbits",
                            dest="nbits",
                            type=int,
                            help="...",
                            default=20)
    fit_parser.add_argument("-n", "--n_discretization-levels",
                            dest="n",
                            type=int,
                            help="...",
                            default=100)
    fit_parser.add_argument("-s", "--label-size",
                            dest="label_size",
                            type=int,
                            help="...",
                            default=5)
    fit_parser.add_argument("--min-subarray-size",
                            dest="min_subarray_size",
                            type=int,
                            help="...",
                            default=3)
    fit_parser.add_argument("--max-subarray-size",
                            dest="max_subarray_size",
                            type=int,
                            help="...",
                            default=15)

    # predict commands
    predict_parser = subparsers.add_parser('predict',
                                           help='Predict commands',
                                           parents=[base_parser],
                                           formatter_class=DefaultsRawDescriptionHelpFormatter)
    predict_parser.set_defaults(which='predict')

    return parser

if __name__ == "__main__":
    prog_name = 'CasLociPredictor'
    parser = argparse_setup(description, epilog)
    args = parser.parse_args()

    if args.no_logging:
        configure_logging(logger, verbosity=args.verbosity)
    else:
        configure_logging(logger, verbosity=args.verbosity, filename=prog_name + '.log')

    logger.debug('-' * 80)
    logger.debug('Program: %s' % prog_name)
    logger.debug('Called with parameters:\n %s' % serialize_dict(args.__dict__))

    start_time = time()
    try:
        main_dipatch(args)
    except Exception:
        import datetime
        curr_time = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
        logger.exception("Program run failed on %s" % curr_time)
    finally:
        end_time = time()
        logger.info('Elapsed time: %.1f sec', end_time - start_time)
