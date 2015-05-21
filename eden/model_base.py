#!/usr/bin/env python

import sys
import os
import random
import re
from time import time, clock
import multiprocessing as mp
import numpy as np
from itertools import tee, chain
from collections import defaultdict

import argparse
import logging
import logging.handlers
from eden.util import configure_logging
from eden.util import serialize_dict


from numpy.random import randint
from numpy.random import uniform

from sklearn.linear_model import SGDClassifier
from sklearn import metrics

from eden.graph import Vectorizer
from eden.util import save_output, store_matrix
from eden.converter.graph.node_link_data import node_link_data_to_eden

class ModelInitializerBase(object):

    def __init__(self):
        pass

    def load_data(self, args):
        iterator = node_link_data_to_eden(args.input_file)
        return iterator

    def load_positive_data(self, args):
        return self.load_data(args.positive_input_file)

    def load_negative_data(self, args):
        return self.load_data(args.negative_input_file)

    def pre_processor_init(self, n_iter):
        def pre_processor(graphs, **args):
            return graphs
        pre_processor_parameters = {}
        return pre_processor, pre_processor_parameters

    def vectorizer_init(self, n_iter):
        vectorizer = Vectorizer()
        vectorizer_parameters = {'complexity': [2, 3, 4]}
        return vectorizer, vectorizer_parameters

    def estimator_init(self, n_iter):
        estimator = SGDClassifier(average=True, class_weight='auto', shuffle=True)
        estimator_parameters = {'n_iter': randint(5, 200, size=n_iter),
                                'penalty': ['l1', 'l2', 'elasticnet'],
                                'l1_ratio': uniform(0.1, 0.9, size=n_iter),
                                'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                                'power_t': uniform(0.1, size=n_iter),
                                'alpha': [10 ** x for x in range(-8, 0)],
                                'eta0': [10 ** x for x in range(-4, -1)],
                                'learning_rate': ["invscaling", "constant", "optimal"],
                                'n_jobs': [-1]}
        return estimator, estimator_parameters

    def add_arguments(self, parser):
        parser.add_argument('--version', action='version', version='0.1')
        return parser

    def add_arguments_fit(self, parser):
        parser.add_argument("-p", "--positive-input-file",
                            dest="positive_input_file",
                            help="Path tofile containing input for the positive class.",
                            required=True)
        parser.add_argument("-n", "--negative-input-file",
                            dest="negative_input_file",
                            help="Path to file containing input for the negative class.",
                            required=True)
        return parser

    def add_arguments_estimate(self, parser):
        return self.add_arguments_fit(parser)

    def add_arguments_base(self, parser):
        parser.add_argument("-i", "--input-file",
                            dest="input_file",
                            help="Path to file containing input.",
                            required=True)
        return parser

    def add_arguments_matrix(self, parser):
        return parser

    def add_arguments_predict(self, parser):
        return parser

    def add_arguments_feature(self, parser):
        return parser


def main_fit(model_initializer, args):
    # init
    pos_train_iterator = model_initializer.load_positive_data(args)
    neg_train_iterator = model_initializer.load_negative_data(args)
    pre_processor, pre_processor_parameters = model_initializer.pre_processor_init(args.n_iter)
    vectorizer, vectorizer_parameters = model_initializer.vectorizer_init(args.n_iter)
    estimator, estimator_parameters = model_initializer.estimator_init(args.n_iter)

    from eden.model import ActiveLearningBinaryClassificationModel
    model = ActiveLearningBinaryClassificationModel(pre_processor=pre_processor,
                                                    estimator=estimator,
                                                    vectorizer=vectorizer,
                                                    fit_vectorizer=args.fit_vectorizer,
                                                    n_jobs=args.n_jobs,
                                                    n_blocks=args.n_blocks,
                                                    block_size=args.block_size,
                                                    pre_processor_n_jobs=args.pre_processor_n_jobs,
                                                    pre_processor_n_blocks=args.pre_processor_n_blocks,
                                                    pre_processor_block_size=args.pre_processor_block_size,
                                                    random_state=args.random_state)
    # save model
    if not os.path.exists(args.output_dir_path):
        os.mkdir(args.output_dir_path)
    full_out_file_name = os.path.join(args.output_dir_path, args.model_file)

    # hyper parameters optimization
    model.optimize(pos_train_iterator, neg_train_iterator,
                   model_name=full_out_file_name,
                   n_iter=args.n_iter,
                   n_inner_iter_estimator=args.n_inner_iter_estimator,
                   pre_processor_parameters=pre_processor_parameters,
                   vectorizer_parameters=vectorizer_parameters,
                   estimator_parameters=estimator_parameters,
                   n_active_learning_iterations=args.n_active_learning_iterations,
                   size_positive=args.size_positive,
                   size_negative=args.size_negative,
                   lower_bound_threshold_positive=args.lower_bound_threshold_positive,
                   upper_bound_threshold_positive=args.upper_bound_threshold_positive,
                   lower_bound_threshold_negative=args.lower_bound_threshold_negative,
                   upper_bound_threshold_negative=args.upper_bound_threshold_negative,
                   max_total_time=args.max_total_time,
                   cv=args.cv,
                   scoring=args.scoring,
                   score_func=lambda u, s: u - s,
                   two_steps_optimization=args.two_steps_optimization)


def main_estimate(model_initializer, args):
    pos_test_iterator = model_initializer.load_positive_data(args)
    neg_test_iterator = model_initializer.load_negative_data(args)

    from eden.model import ActiveLearningBinaryClassificationModel
    model = ActiveLearningBinaryClassificationModel()
    model.load(args.model_file)
    logger.info(model.get_parameters())
    apr, rocauc = model.estimate(pos_test_iterator, neg_test_iterator)


def main_predict(model_initializer, args):
    iterator = model_initializer.load_data(args)
    from itertools import tee
    iterator, iterator_ = tee(iterator)

    from eden.model import ActiveLearningBinaryClassificationModel
    model = ActiveLearningBinaryClassificationModel()
    model.load(args.model_file)
    logger.info(model.get_parameters())
    
    predictions = model.decision_function(iterator)
    text = []
    for p in predictions:
        text.append(str(p) + "\n")
    save_output(text=text, output_dir_path=args.output_dir_path, out_file_name='predictions.txt')
    
    text = []
    for p in predictions:
        if p > 0:
            prediction = 1
        else:
            prediction = -1
        text.append(str(prediction) + "\n")
    save_output(text=text, output_dir_path=args.output_dir_path, out_file_name='classifications.txt')
    
    text = []
    from itertools import izip
    info_iterator = model.get_info(iterator_)
    for p,info in izip(predictions,info_iterator):
        text.append("%.4f\t%s\n"%(p,info))
    save_output(text=text, output_dir_path=args.output_dir_path, out_file_name='info.txt')
    

def main_matrix(model_initializer, args):
    iterator = model_initializer.load_data(args)

    from eden.model import ActiveLearningBinaryClassificationModel
    model = ActiveLearningBinaryClassificationModel()
    model.load(args.model_file)
    logger.info(model.get_parameters())
    X = model._data_matrix(iterator)
    K = metrics.pairwise.pairwise_kernels(X, metric='linear')
    store_matrix(matrix=K, output_dir_path=args.output_dir_path, out_file_name='Gram_matrix', output_format=args.output_format)


def main_feature(model_initializer, args):
    iterator = model_initializer.load_data(args)

    from eden.model import ActiveLearningBinaryClassificationModel
    model = ActiveLearningBinaryClassificationModel()
    model.load(args.model_file)
    logger.info(model.get_parameters())
    X = model._data_matrix(iterator)
    store_matrix(matrix=X, output_dir_path=args.output_dir_path, out_file_name='data_matrix', output_format=args.output_format)


def main(model_initializer, args):
    if args.which == 'fit':
        main_fit(model_initializer, args)
    elif args.which == 'estimate':
        main_estimate(model_initializer, args)
    elif args.which == 'predict':
        main_predict(model_initializer, args)
    elif args.which == 'matrix':
        main_matrix(model_initializer, args)
    elif args.which == 'feature':
        main_feature(model_initializer, args)
    else:
        raise Exception('Unknown mode: %s' % args.which)


def argparse_setup(model_initializer, DESCRIPTION, EPILOG):
    class DefaultsRawDescriptionHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        # To join the behaviour of RawDescriptionHelpFormatter with that of ArgumentDefaultsHelpFormatter
        pass

    parser = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG, formatter_class=DefaultsRawDescriptionHelpFormatter)
    parser = model_initializer.add_arguments(parser)
    parser.add_argument("-v", "--verbosity",
                        action="count",
                        help="Increase output verbosity")
    parser.add_argument("-x", "--no-logging",
                        dest="no_logging",
                        help="If set, do not log on file.",
                        action="store_true")

    subparsers = parser.add_subparsers(help='commands')
    # fit commands
    fit_parser = subparsers.add_parser('fit', help='Fit commands', formatter_class=DefaultsRawDescriptionHelpFormatter)
    fit_parser.set_defaults(which='fit')
    # add domain specific arguments
    fit_parser = model_initializer.add_arguments_fit(fit_parser)
    fit_parser.add_argument("-o", "--output-dir",
                            dest="output_dir_path",
                            help="Path to output directory.",
                            default="out")
    fit_parser.add_argument("-m", "--model-file",
                            dest="model_file",
                            help="Model file name. Note: it will be located in the output directory.",
                            default="model")
    fit_parser.add_argument("-e", "--n-iter",
                            dest="n_iter",
                            type=int,
                            help="Number of randomly generated hyper parameter configurations tried during the discriminative model optimization. A value of 1 implies using the estimator default values.",
                            default=20)
    fit_parser.add_argument("--n-inner-iter-estimator",
                            dest="n_inner_iter_estimator",
                            type=int,
                            help="Number of randomly generated hyper parameter configurations tried for the estimator for each parameter configuration of the pre-processor and vectorizer during optimization.",
                            default=5)
    fit_parser.add_argument("--n-active-learning-iterations",
                            dest="n_active_learning_iterations",
                            type=int,
                            help="Number of iterations in the active lerning cycle. A value of 0 means to avoid active learning.",
                            default=0)
    fit_parser.add_argument("--size-positive",
                            dest="size_positive",
                            type=int,
                            help="Number of positive instances that have to be sampled in the active lerning cycle. A value of -1 means to use all instances, i.e. not to use active learning for the positive instances.",
                            default=-1)
    fit_parser.add_argument("--size-negative",
                            dest="size_negative",
                            type=int,
                            help="Number of negative instances that have to be sampled in the active lerning cycle. A value of -1 means to use all instances, i.e. not to use active learning for the negative instances.",
                            default=-1)
    fit_parser.add_argument("--lower-bound-threshold-positive",
                            dest="lower_bound_threshold_positive",
                            type=int,
                            help="Value of the score threshold to determine when to accept positive instances: positive instances with a score higher than the specified value will be accepted as candidates.",
                            default=-1)
    fit_parser.add_argument("--lower-bound-threshold-negative",
                            dest="lower_bound_threshold_negative",
                            type=int,
                            help="Value of the score threshold to determine when to accept negative instances: negative instances with a score higher than the specified value will be accepted as candidates.",
                            default=-1)
    fit_parser.add_argument("--upper-bound-threshold-positive",
                            dest="upper_bound_threshold_positive",
                            type=int,
                            help="Value of the score threshold to determine when to accept positive instances: positive instances with a score lower than the specified value will be accepted as candidates.",
                            default=1)
    fit_parser.add_argument("--upper-bound-threshold-negative",
                            dest="upper_bound_threshold_negative",
                            type=int,
                            help="Value of the score threshold to determine when to accept negative instances: negative instances with a score lower than the specified value will be accepted as candidates.",
                            default=1)
    fit_parser.add_argument("--fit-vectorizer",
                            dest="fit_vectorizer",
                            help="If set, activate the fitting procedure for the vectorizer on positive instances only.",
                            action="store_true")
    fit_parser.add_argument("--max-total-time",
                            dest="max_total_time",
                            type=int,
                            help="Maximal number of seconds for the duration of the optimization phase. After that the procedure is forcefully stopped. A value of -1 means no time limit.",
                            default=-1)
    fit_parser.add_argument("--two-steps-optimization",
                            dest="two_steps_optimization",
                            help="If set, activate a refinement procedure anfter n_iter/2 steps that samples only among the parameters that have previously improved the results.",
                            action="store_true")
    fit_parser.add_argument("--scoring", choices=['accuracy', 'roc_auc', 'average_precision', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'log_loss', 'precision', 'recall'],
                            help="The scoring strategy for evaluating in cross validation the quality of a hyper parameter combination.",
                            default='roc_auc')
    fit_parser.add_argument("--cv",
                            type=int,
                            help="Cross validation size.",
                            default=10)
    fit_parser.add_argument("-B", "--nbits",
                            type=int,
                            help="Number of bits used to express the graph kernel features. A value of 20 corresponds to 2**20=1 million possible features.",
                            default=20)
    fit_parser.add_argument("-j", "--n-jobs",
                            dest="n_jobs",
                            type=int,
                            help="Number of cores to use in multiprocessing.",
                            default=4)
    fit_parser.add_argument("-k", "-block-size",
                            dest="block_size",
                            type=int,
                            help="Number of instances per block for the multiprocessing elaboration.",
                            default=None)
    fit_parser.add_argument("-b", "--n-blocks",
                            dest="n_blocks",
                            type=int,
                            help="Number of blocks in which to divide the input for the multiprocessing elaboration.",
                            default=10)
    fit_parser.add_argument("--pre-processor-n-jobs",
                            dest="pre_processor_n_jobs",
                            type=int,
                            help="Number of cores to use in multiprocessing.",
                            default=4)
    fit_parser.add_argument("--pre-processor-n-blocks",
                            dest="pre_processor_n_blocks",
                            type=int,
                            help="Number of blocks in which to divide the input for the multiprocessing elaboration.",
                            default=10)
    fit_parser.add_argument("--pre-processor-block-size",
                            dest="pre_processor_block_size",
                            type=int,
                            help="Number of instances per block for the multiprocessing elaboration.",
                            default=None)
    fit_parser.add_argument("-r", "--random-state",
                            dest="random_state",
                            type=int,
                            help="Random seed.",
                            default=1)

    # estimate commands
    estimate_parser = subparsers.add_parser('estimate', help='Estimate commands', formatter_class=DefaultsRawDescriptionHelpFormatter)
    estimate_parser.set_defaults(which='estimate')
    estimate_parser = model_initializer.add_arguments_estimate(estimate_parser)
    estimate_parser.add_argument("-m", "--model-file",
                                 dest="model_file",
                                 help="Path to a fit model file.",
                                 required=True)
    estimate_parser.add_argument("-o", "--output-dir",
                                 dest="output_dir_path",
                                 help="Path to output directory.",
                                 default="out")

    # base parser
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser = model_initializer.add_arguments_base(base_parser)
    base_parser.add_argument("-m", "--model-file",
                             dest="model_file",
                             help="Path to a fit model file.",
                             default="model")
    base_parser.add_argument("-o", "--output-dir",
                             dest="output_dir_path",
                             help="Path to output directory.",
                             default="out")

    # predict commands
    predict_parser = subparsers.add_parser('predict',
                                           help='Predict commands',
                                           parents=[base_parser],
                                           formatter_class=DefaultsRawDescriptionHelpFormatter)
    predict_parser.set_defaults(which='predict')
    predict_parser = model_initializer.add_arguments_predict(predict_parser)

    # matrix commands
    matrix_parser = subparsers.add_parser('matrix',
                                          help='Matrix commands',
                                          parents=[base_parser],
                                          formatter_class=DefaultsRawDescriptionHelpFormatter)
    matrix_parser.set_defaults(which='matrix')
    matrix_parser = model_initializer.add_arguments_matrix(matrix_parser)
    matrix_parser.add_argument("-t", "--output-format",  choices=["text", "numpy", "MatrixMarket", "joblib"],
                               dest="output_format",
                               help="Output file format.",
                               default="MatrixMarket")

    # feature commands
    feature_parser = subparsers.add_parser('feature',
                                           help='Feature commands',
                                           parents=[base_parser],
                                           formatter_class=DefaultsRawDescriptionHelpFormatter)
    feature_parser.set_defaults(which='feature')
    feature_parser = model_initializer.add_arguments_feature(feature_parser)
    feature_parser.add_argument("-t", "--output-format",  choices=["text", "numpy", "MatrixMarket", "joblib"],
                                dest="output_format",
                                help="Output file format.",
                                default="MatrixMarket")
    return parser


def main_script(model_initializer=None, description=None, epilog=None, prog_name=None, logger=None):
    parser = argparse_setup(model_initializer, description, epilog)
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
        main(model_initializer, args)
    except Exception:
        import datetime
        curr_time = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
        logger.exception("Program run failed on %s" % curr_time)
    finally:
        end_time = time()
        logger.info('Elapsed time: %.1f sec', end_time - start_time)
