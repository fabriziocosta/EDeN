from eden.modifier.seq import seq_to_seq, shuffle_modifier
from eden.sequence import Vectorizer
from eden.util import fit, predict
import numpy as np
from itertools import combinations_with_replacement
import logging
logger = logging.getLogger(__name__)


class SequenceGenerator(object):

    def __init__(self,
                 n_differences=1,
                 enhance=True,
                 vectorizer=Vectorizer(complexity=3),
                 n_jobs=-1,
                 random_state=1):
        """Generate sequences starting from input sequences that are 'better' if enhance is set to True
        ('worse' otherwise) given the set of sequences used in the fit phase.

        Parameters
        ----------
        n_differences : int (default 1)
            Number of characters that differ for the generated sequence from the original input sequence.

        enhance : bool (default True)
            If set to True then the score computed by the estimator will be higher for the sequences
            generated than for the input sequences. If False than the score will be lower.

        vectorizer : EDeN sequence vectorizer
            The vectorizer to map sequences to sparse vectors.

        n_jobs : int (default -1)
            The number of cores to use in parallel. -1 indicates all available.

        random_state: int (default 1)
            The random seed.
        """

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_differences = n_differences
        self.enhance = enhance
        self.vectorizer = vectorizer
        self.estimator = None

    def fit(self, pos_seqs, neg_seqs=None, times=2, order=2):
        """Fit an estimator to discriminate the pos_seqs from the neg_seqs.

        Parameters
        ----------
        pos_seqs : iterable strings
            Input sequences.

        neg_seqs : iterable strings (default: None)
            If not None the program uses these as negative examples. If
            it is None, then negative sequences are generated as random
            shuffling of the positive sequences.

        times: int (default: 2)
            Factor between number of negatives and number of positives.

        order: int (default: 2)
            Size of the minimum block to shuffle: 1 means shuffling single characters,
            2 means shuffling pairs of characters, etc.

        Returns
        -------
        self.
        """

        if neg_seqs is None:
            neg_seqs = list(seq_to_seq(pos_seqs, modifier=shuffle_modifier, times=times, order=order))
        self.estimator = fit(pos_seqs, neg_seqs, self.vectorizer,
                             n_jobs=self.n_jobs,
                             cv=10,
                             n_iter_search=1,
                             random_state=self.random_state,
                             n_blocks=5,
                             block_size=None)
        return self

    def sample(self, seqs, n_seqs=1, show_score=False, enhance=None, n_differences=None):
        """Generate sequences starting from input sequences that are 'better' if enhance is set to True
        ('worse' otherwise) given the set of sequences used in the fit phase.

        Parameters
        ----------
        seqs : iterable strings
            Input sequences.

        n_seqs : int (default: 1)
            Number of sequences to be generated starting from each sequence in input.

        show_score: bool (default: False)
            If True the return type is a pair consisting of a score and a sequence. If
            False the return type is a sequence.

        enhance : bool (default None)
            If set to True then the score computed by the estimator will be higher for the sequences
            generated than for the input sequences. If False than the score will be lower. If None
            the state set in the initializer is used.

        n_differences : int (default None)
            Number of characters that differ for the generated sequence from the original input sequence.
            If None the number set in the initializer is used.

        Returns
        -------
        sequences : iterable sequences
            List of sequences or (score, sequence) pairs if show_score is True.
        """
        if enhance is not None:
            self.enhance = enhance
        if n_differences is not None:
            self.n_differences = n_differences
        for seq in seqs:
            if show_score:
                preds = predict(iterable=[seq],
                                estimator=self.estimator,
                                vectorizer=self.vectorizer,
                                mode='decision_function', n_blocks=5, block_size=None, n_jobs=self.n_jobs)
                logger.debug('%s\n%+.3f %s' % (seq[0], preds[0], seq[1]))
            gen_seqs = self._generate(seq, n_seqs=n_seqs, show_score=show_score)
            for gen_seq in gen_seqs:
                yield gen_seq

    def _generate(self, input_seq, n_seqs=1, show_score=False):
        header, seq = input_seq
        # find best/worst n_differences positions
        seq_items, n_differences_ids = self._find_key_positions(seq)
        # replace all possible kmers of size n_differences
        gen_seqs = list(self._replace(seq_items, n_differences_ids))
        # keep the best/worst
        preds = predict(iterable=gen_seqs,
                        estimator=self.estimator,
                        vectorizer=self.vectorizer,
                        mode='decision_function', n_blocks=5, block_size=None, n_jobs=self.n_jobs)
        sorted_pred_ids = np.argsort(preds)
        if self.enhance:
            n_seqs_ids = sorted_pred_ids[-n_seqs:]
            n_seqs_ids = n_seqs_ids[::-1]
        else:
            n_seqs_ids = sorted_pred_ids[:n_seqs]
        if show_score:
            return zip(np.array(preds)[n_seqs_ids], np.array(gen_seqs)[n_seqs_ids])
        else:
            return np.array(gen_seqs)[n_seqs_ids]

    def _replace(self, seq_items, n_differences_ids):
        alphabet = set(seq_items)
        kmers = combinations_with_replacement(alphabet, self.n_differences)
        for kmer in kmers:
            curr_seq = seq_items
            for i, symbol in enumerate(kmer):
                pos = n_differences_ids[i]
                curr_seq[pos] = symbol
            gen_seq = ''.join(curr_seq)
            yield gen_seq

    def _find_key_positions(self, seq):
        # annotate seq using estimator
        annotation = self.vectorizer.annotate([seq], estimator=self.estimator)
        seq_items, scores = annotation.next()
        assert(len(seq_items) == len(seq))
        assert(len(scores) == len(seq))
        sorted_ids = np.argsort(scores)
        if self.enhance:
            n_differences_ids = sorted_ids[:self.n_differences]
        else:
            n_differences_ids = sorted_ids[-self.n_differences:]
        return seq_items, n_differences_ids
