import argparse
import logging

from eden.util.globals import EPILOG

def setup(description, setup_parameters_func):
	parser = argparse.ArgumentParser(description=description,
		epilog=EPILOG,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	setup_parameters_func(parser)
	args = parser.parse_args()

	log_level = logging.INFO
	if args.verbosity == 1:
		log_level = logging.WARNING
		print "WARNING"
	elif args.verbosity >= 2:
		log_level = logging.DEBUG
		print "DEBUG"
	logging.basicConfig(filename='log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=log_level)

	return args