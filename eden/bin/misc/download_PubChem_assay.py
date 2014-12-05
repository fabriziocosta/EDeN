#!/usr/bin/env python

import sys
import os
import requests
import argparse

DESCRIPTION = """
Download molecular sets from PubChem.
"""

PROLOG='https://pubchem.ncbi.nlm.nih.gov/rest/pug/'

def get_compound_set(fname, size, listkey):
    with open(fname,'w') as file_handle:
        stepsize=50
        index_start=0
        for chunk, index_end in enumerate(range(0,size+stepsize,stepsize)):
            if index_end is not 0 :
                print 'Chunk %s) Processing compounds %s to %s (of a total of %s)' % (chunk, index_start, index_end-1, size)
                RESTQ = PROLOG + 'compound/listkey/' + str(listkey) + '/SDF?&listkey_start=' + str(index_start) + '&listkey_count=' + str(stepsize)
                reply=requests.get(RESTQ)
                file_handle.write(reply.text)
            index_start = index_end
        print('Compounds downloaded in file: %s' % fname)


def get_remote_data(RESTQ, args, out_file_name):
	reply=requests.get(RESTQ)
	listkey = reply.json()['IdentifierList']['ListKey']
	size = reply.json()['IdentifierList']['Size'] 
	if not os.path.exists(args.output_dir_path) :
		os.mkdir(args.output_dir_path)
	full_out_file_name = os.path.join(args.output_dir_path, out_file_name)
	get_compound_set(fname = full_out_file_name, size = size, listkey = listkey)


def main(args):
    AID=args.assay_id
    #active
    RESTQ = PROLOG + 'assay/aid/' + AID + '/cids/JSON?cids_type=active&list_return=listkey'
    out_file_name = 'AID' + AID + '_active.sdf'
    get_remote_data(RESTQ, args, out_file_name)
    
    #inactive
    RESTQ = PROLOG + 'assay/aid/' + AID + '/cids/JSON?cids_type=inactive&list_return=listkey'
    out_file_name = 'AID' + AID + '_inactive.sdf'
    get_remote_data(RESTQ, args, out_file_name)
    

if __name__  == "__main__":
	parser = argparse.ArgumentParser(description = DESCRIPTION, formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-i", "--assay-id",  
	dest = "assay_id",
	help = "Numerical ID of assay in PubChem.", 
	required = True)
	parser.add_argument("-o", "--output-dir", 
	dest = "output_dir_path", 
	help = "Path to output directory.",
	default = "out")
	args = parser.parse_args()
	main(args)
