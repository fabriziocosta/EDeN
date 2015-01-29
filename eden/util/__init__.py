import numpy as np
from scipy import io
from sklearn.externals import joblib
import requests
import os

def read( uri ):
    """
    Abstract read function. EDeN can accept a URL, a file path and a python list.
    In all cases an iteratatable object should be returned.
    """
    if hasattr(uri, '__iter__'):
        return uri
    else:
        try:
            # try if it is a URL and we can open it
            f = requests.get( uri ).text.split('\n')
        except ValueError:
            # Assume it is a file object
            f = open( uri )
        return f

def load_target( name ):
    """
    Return a numpy array of integers to be used as target vector.

    Parameters
    ----------
    name : string
        A pointer to the data source.

    """

    Y = [ y.strip() for y in read( name ) if y ]
    return np.array(Y).astype(int)


def store_matrix(matrix = '', output_dir_path = '', out_file_name = '', output_format = ''):
    if not os.path.exists(output_dir_path) :
        os.mkdir(output_dir_path)
    full_out_file_name = os.path.join(output_dir_path, out_file_name)
    if output_format == "MatrixMarket":
        if len(matrix.shape) == 1:
            raise Exception("'MatrixMarket' format supports only 2D dimensional array and not vectors")
        else:
            io.mmwrite(full_out_file_name, matrix, precision = None)
    elif output_format == "numpy":
        np.save(full_out_file_name, matrix)
    elif output_format == "joblib":
        joblib.dump(matrix, full_out_file_name)
    elif output_format == "text":
        with open(full_out_file_name, "w") as f:
            if len(matrix.shape) == 1:
                for x in matrix:
                    f.write("%s\n"%(x))
                #data_str = map(str, matrix)
                #f.write('\n'.join(data_str))
            else:
                raise Exception("Currently 'text' format supports only mono dimensional array and not matrices")


def dump(obj, output_dir_path = '', out_file_name = ''):
    if not os.path.exists(output_dir_path) :
        os.mkdir(output_dir_path)
    full_out_file_name = os.path.join(output_dir_path, out_file_name) + ".pkl"
    joblib.dump(obj, full_out_file_name) 

def load(output_dir_path = '', out_file_name = ''):
    if not os.path.exists(output_dir_path) :
        os.mkdir(output_dir_path)
    full_out_file_name = os.path.join(output_dir_path, out_file_name) + ".pkl"
    obj=joblib.load(full_out_file_name) 
    return obj