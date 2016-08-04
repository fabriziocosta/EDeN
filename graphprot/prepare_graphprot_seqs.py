#!/usr/bin/env python

# draft implementation
# * TODO:
#   * centering should be optional
#   * viewpoint should be optional
#   * check for nonunique ids and warn
#   * check for bedtools version
#   * write bed files for sequence coordinates
#   * set rnd init for shuffling to have reproducible results
#   * use my own temporary sequence files, properly clean up afterwards
#   * check if seq length and core length arguments match or handle properly
#   * handle input/output error gracefully
#   * check if input bed coordinates are stranded
#   * seq_len == core_len does not work
#   * have to set default output prefix
#   * negatives may be shuffled onto the wrong strand. investigate if this is a recent bedtools thing. otherwise shuffle separately for both strands
#   * handle "crazy" bed formats (eg 7 fields from piranha)

from __future__ import print_function
import argparse
from csv import reader
from itertools import izip
import logging
from eden.util import configure_logging
from pybedtools.featurefuncs import midpoint
from pybedtools.helpers import get_chromsizes_from_ucsc
from pybedtools import BedTool

# parse command line arguments
# positional arguments
parser = argparse.ArgumentParser(
    description="Create coordinates and fasta sequences for use with GraphProt.")
parser.add_argument(
    "bsites_fn", help="Path to binding site coordiantes in bed format")
parser.add_argument(
    "genome_id", help="Genome UCSC id")
parser.add_argument(
    "genome_fa_fn", help="Genome fasta sequences")
# optional arguments
parser.add_argument(
    "--seq_length",
    type=int,
    default=150,
    help="Length of sequences to create")
parser.add_argument(
    "--core_length",
    type=int,
    default=48,
    help="Length of viewpoint region at center of sequence")
parser.add_argument(
    "--output_file_prefix",
    default="",
    help="Prefix to use for output filenames")
parser.add_argument(
    "--chromosome_limits",
    help="Path to file containing chromosome limites as required by bedtools. Use this parameter disables automatic lookup via the genome id.")
parser.add_argument(
    "--negative_site_candidate_regions_fn",
    help="Path to regions considered for placement of negatives in bed format")
parser.add_argument(
    "-v", "--verbosity",
    action="count",
    help="Increase output verbosity")
args = parser.parse_args()

logger = logging.getLogger()
configure_logging(logger, verbosity=args.verbosity)

# fixed global variables
npeek = 2

# check chromsizes retreival
if (args.chromosome_limits is None):
    # check if genome_id can be found,
    chromsizes = get_chromsizes_from_ucsc(args.genome_id)
    logging.debug("Number of chromosomes: {}.".format(len(chromsizes)))
    # otherwise request manual definition of chromosome limits
    if (len(chromsizes) == 0):
        logging.error("Error: retrieving chromosome sizes from UCSC failed. Please specify manually using parameter --chromosome_limits")
        exit(1)

# output file arguments
pos_core_bed_fn = args.output_file_prefix + ".positives_core.bed"
neg_core_bed_fn = args.output_file_prefix + ".negatives_core.bed"
# TODO: use
pos_seq_bed_fn = args.output_file_prefix + ".positives_seq.bed"
# TODO: use
neg_seq_bed_fn = args.output_file_prefix + ".negatives_seq.bed"
pos_seq_fa_fn = args.output_file_prefix + ".positives.fa"
neg_seq_fa_fn = args.output_file_prefix + ".negatives.fa"

# calculate flank lengths
flank_length = args.seq_length - args.core_length
flank_upstream_length = int(flank_length / 2)
flank_downstream_length = int(flank_length / 2) + (flank_length % 2)
if (args.core_length + flank_upstream_length + flank_downstream_length != args.seq_length):
    raise Exception("Error: bad length calculation.")


def dbg_head(sites, description="", n=npeek, run=True):
    """Print the first few bed entries."""
    if run:
        logging.debug(description)
        for i in sites[0:n]:
            logging.debug(i)


def prefix_neg(feature, prefix="negative_from_"):
    """Modify BedTool feature by adding a prefix."""
    feature.name = prefix + feature.name
    return feature


def offset_zero_by_one(feature):
    """Sets the start coordinate to 1 if it is actually 0.

    Required for the flanking to work properly in those cases.
    """
    if feature.start == 0:
        feature.start += 1
    return feature


def get_flanks(cores,
               flank_upstream_length=flank_upstream_length,
               flank_downstream_length=flank_downstream_length):
    """Calculate flanking regions of a core region."""
    if args.chromosome_limits is not None:
        logging.debug("using chromosome_limits " + args.chromosome_limits)
        # get upstream flanks
        flanks_upstream = cores.flank(
            s=True,
            l=flank_upstream_length,
            r=0,
            g=args.chromosome_limits).saveas()
        # get downstream flanks
        flanks_downstream = cores.flank(
            s=True,
            r=flank_downstream_length,
            l=0,
            g=args.chromosome_limits).saveas()
    else:
        # get upstream flanks
        flanks_upstream = cores.flank(
            s=True,
            l=flank_upstream_length,
            r=0,
            genome=args.genome_id).saveas()
        # get downstream flanks
        flanks_downstream = cores.flank(
            s=True,
            r=flank_downstream_length,
            l=0,
            genome=args.genome_id).saveas()
    # check if sites and flanks have the same number of entries
    if cores.count() == flanks_upstream.count() == flanks_downstream.count():
        return flanks_upstream, flanks_downstream
    else:
        if True:
            cores.saveas("debug_cores.bed")
            flanks_upstream.saveas("debug_upstream.bed")
            flanks_downstream.saveas("debug_downstream.bed")
        else:
            cores.saveas()
            flanks_upstream.saveas()
            flanks_downstream.saveas()
        raise Exception("Error: numbers of cores and flanks don't match: got " + str(cores.count()) + " cores, " + str(
            flanks_upstream.count()) + " upstream flanks and " + str(flanks_downstream.count()) + " downstream flanks.")


def get_seqs(cores,
             flanks_upstream,
             flanks_downstream,
             viewpointfa_fn,
             genome_fa_fn=args.genome_fa_fn):
    """Prepare sequences and write them to disk."""
    # get sequences
    genome_fa = BedTool(genome_fa_fn)
    cores = cores.sequence(
        fi=genome_fa,
        s=True,
        tab=True, name=True).save_seqs(cores.fn + ".tabseq")
    flanks_upstream = flanks_upstream.sequence(
        fi=genome_fa,
        s=True,
        tab=True,
        name=True).save_seqs(flanks_upstream.fn + ".tabseq")
    flanks_downstream = flanks_downstream.sequence(
        fi=genome_fa,
        s=True,
        tab=True,
        name=True).save_seqs(flanks_downstream.fn + ".tabseq")
    # write sequences to disk
    fup_seq_fn = flanks_upstream.seqfn
    cores_seq_fn = cores.seqfn
    fdown_seq_fn = flanks_downstream.seqfn
    viewpointfa = open(viewpointfa_fn, "wb")
    with open(fup_seq_fn, "rb") as fup_tabseq, open(cores_seq_fn, "rb") as core_tabseq, open(fdown_seq_fn, "rb") as fdown_tabseq:
        fup_reader = reader(fup_tabseq, delimiter="\t")
        core_reader = reader(core_tabseq, delimiter="\t")
        fdown_reader = reader(fdown_tabseq, delimiter="\t")
        for fup, core, fdown in izip(fup_reader, core_reader, fdown_reader):
            assert fup[0] == core[0] == fdown[0], "Error: sequence ids of cores and flanks don't match."
            # setup fasta headers and sequences
            fa_header = ">" + core[0]
            seq_viewpoint = fup[1].lower() + core[1].upper() + fdown[1].lower()
            # seq_normal = fup[1].upper() + core[1].upper() + fdown[1].upper()

            viewpointfa.write(fa_header + "\n")
            viewpointfa.write(seq_viewpoint + "\n")
    viewpointfa.close()

# prepare input coordinates
bsites = BedTool(args.bsites_fn).sort().saveas()
centers = bsites.each(midpoint).saveas()

# prepare positive instances
logging.info("preparing positive instances")
if (args.chromosome_limits):
    logging.debug("using chromosome_limits " + args.chromosome_limits)
    cores = centers.slop(s=True,
                         l=int(args.core_length / 2),
                         # -1 to account for the center nucleotide!
                         r=int(args.core_length / 2) +
                         (args.core_length % 2) - 1,
                         g=args.chromosome_limits).each(offset_zero_by_one).saveas(pos_core_bed_fn)
else:
    cores = centers.slop(s=True,
                         l=int(args.core_length / 2),
                         # -1 to account for the center nucleotide!
                         r=int(args.core_length / 2) +
                         (args.core_length % 2) - 1,
                         genome=args.genome_id).each(offset_zero_by_one).saveas(pos_core_bed_fn)

flanks_upstream, flanks_downstream = get_flanks(cores)
get_seqs(cores, flanks_upstream, flanks_downstream, pos_seq_fa_fn)

# prepare negative sites if requested
if args.negative_site_candidate_regions_fn:
    # get negative candidate regions
    negative_site_candidate_regions = BedTool(
        args.negative_site_candidate_regions_fn)
    # remove input binding sites from negative candidate regions
    processed_negative_site_candidate_regions = negative_site_candidate_regions.subtract(
        bsites,
        s=True).saveas()

    # create negative core sites by placing within candidate regions
    logging.info("preparing negative instances")
    logging.info("starting from " + str(cores.count()) + " positive cores")
    if args.chromosome_limits:
        logging.debug("using chromosome_limits " + args.chromosome_limits)
        neg_cores = cores.shuffle(
            g=args.chromosome_limits,
            chrom=True,
            incl=processed_negative_site_candidate_regions.fn,
            noOverlapping=True).each(prefix_neg).saveas(neg_core_bed_fn)
        logging.info("derived negative cores: " + str(neg_cores.count()))
        neg_fup, neg_fdown = get_flanks(neg_cores)
        get_seqs(neg_cores, neg_fup, neg_fdown, neg_seq_fa_fn)
    else:
        neg_cores = cores.shuffle(
            genome=args.genome_id,
            chrom=True,
            incl=processed_negative_site_candidate_regions.fn,
            noOverlapping=True).each(prefix_neg).saveas(neg_core_bed_fn)
        logging.info("derived negative cores: " + str(neg_cores.count()))
        neg_fup, neg_fdown = get_flanks(neg_cores)
        get_seqs(neg_cores, neg_fup, neg_fdown, neg_seq_fa_fn)
