import argparse
import logging
import logging.handlers
from eden.util.globals import EPILOG


def common_arguments(parser):
    """
    Common commandline arguments.
    """
    parser.add_argument("-i", "--input-file",
    	dest = "input_file",
    	help = "Path to your graph file.", 
    	required = True)
    parser.add_argument( "-r","--radius",
        type = int, 
        help = "Size of the largest radius used in EDeN.", 
        default = 4)
    parser.add_argument( "-d", "--distance",
        type = int, 
        help = "Size of the largest distance used in EDeN.", 
        default = 8)
    parser.add_argument( "--min-radius",
        dest = "min_r", 
        type = int, 
        help = "Size of the smallest radius used in EDeN.", 
        default = 0)
    parser.add_argument( "--min-distance",
        dest = "min_d", 
        type = int, 
        help = "Size of the smallest distance used in EDeN.", 
        default = 0)
    parser.add_argument( "-b", "--num-bits",
		dest = "nbits",
		type = int, 
		help = "Number of bits used to encode features, hence the feature space has size x \in IR 2^num-bits .", 
		default = 20)
    parser.add_argument( "-j", "--num-jobs",
		dest = "n_jobs",
		type = int, 
		help = "The number of CPUs to use. -1 means all available CPUs.", 
		default = -1)
    parser.add_argument("-o", "--output-dir", 
		dest = "output_dir_path", 
		help = "Path to output directory.",
		default = "out")
    parser.add_argument("-t", "--output-format",  choices = ["text", "numpy", "MatrixMarket", "joblib"],
    	dest = "output_format",
    	help = "Output file format.", 
    	default = "MatrixMarket")
    parser.add_argument("-v", "--verbosity", 
		action = "count",
		help = "Increase output verbosity")
    return parser


def arguments_parser(description = None, setup_parameters_func = None):
	"""
	Initialize parameters parsing
	"""
	parser = argparse.ArgumentParser(description = description, epilog = EPILOG, formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser = setup_parameters_func(parser)
	args = parser.parse_args()
	return args


def logger(logger_name = None, filename = None, verbosity = 0):
	logger = logging.getLogger(logger_name)
	log_level = logging.WARNING
	if  verbosity == 1:
		log_level = logging.INFO
	elif  verbosity >= 2:
		log_level = logging.DEBUG
	logger.setLevel(logging.DEBUG)
	# create console handler
	ch = logging.StreamHandler()
	ch.setLevel(log_level)
	# create a file handler
	fh = logging.handlers.RotatingFileHandler(filename = filename , maxBytes=100000, backupCount=10)
	fh.setLevel(logging.DEBUG)
	# create formatter
	cformatter = logging.Formatter('%(message)s')
	# add formatter to ch
	ch.setFormatter(cformatter)
	# create formatter
	fformatter = logging.Formatter('%(asctime)s : %(name)-10s : %(levelname)-6s : %(message)s')
	# and to fh
	fh.setFormatter(fformatter)
	# add handlers to logger
	logger.addHandler(ch)
	logger.addHandler(fh)

	return logger