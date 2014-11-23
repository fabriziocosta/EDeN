import argparse
import logging

from eden.util.globals import EPILOG

def setup(description, setup_parameters_func):
	"""
	Initialize parameters parsing and logger
	"""
	parser = argparse.ArgumentParser(description = description, epilog = EPILOG, formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser = setup_parameters_func(parser)
	args = parser.parse_args()
	return args
