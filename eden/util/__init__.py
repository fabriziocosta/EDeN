import numpy as np
from scipy import io
from sklearn.externals import joblib
import requests
import os
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn import cross_validation
from scipy.stats import randint
from scipy.stats import uniform
import numpy as np
from scipy import stats
from scipy.sparse import vstack


def fit_estimator(positive_data_matrix = None, negative_data_matrix = None, target = None, cv = 10, n_jobs = -1):
    assert(positive_data_matrix is not None), 'ERROR: expecting non null positive_data_matrix'
    if target is None and negative_data_matrix is not None:
        yp =  [1] * positive_data_matrix.shape[0]
        yn = [-1] * negative_data_matrix.shape[0]
        y = np.array(yp + yn)
        X = vstack( [positive_data_matrix,negative_data_matrix] , format = "csr")
    if target is not None:
        X = positive_data_matrix
        y = target

    predictor = SGDClassifier(class_weight='auto', shuffle = True, n_jobs = n_jobs)
    #hyperparameter optimization
    param_dist = {"n_iter": randint(5, 100),
                  "power_t": uniform(0.1),
                  "alpha": uniform(1e-08,1e-03),
                  "eta0" : uniform(1e-03,10),
                  "penalty": ["l1", "l2", "elasticnet"],
                  "learning_rate": ["invscaling", "constant","optimal"]}
    scoring = 'roc_auc'
    n_iter_search = 20
    random_search = RandomizedSearchCV( predictor, param_distributions = param_dist, n_iter = n_iter_search, cv = cv, scoring = scoring, n_jobs = n_jobs )
    random_search.fit( X, y )
    optpredictor= SGDClassifier( class_weight='auto', shuffle = True, n_jobs = n_jobs, **random_search.best_params_ )
    #fit the predictor on all available data
    optpredictor.fit( X, y ) 
    
    
    print 'Classifier:'
    print optpredictor
    print '-'*73

    print 'Predictive performance:'
    #assess the generalization capacity of the model via a 10-fold cross validation
    for scoring in ['accuracy','precision', 'recall', 'f1', 'average_precision', 'roc_auc']:
        scores = cross_validation.cross_val_score( optpredictor, X, y, cv = cv, scoring = scoring, n_jobs = n_jobs )
        print( '%20s: %.3f +- %.3f' % ( scoring, np.mean( scores ), np.std( scores ) ) )
    print '-'*73
    
    return optpredictor


def read( uri ):
    """
    Abstract read function. EDeN can accept a URL, a file path and a python list.
    In all cases an iteratatable object should be returned.
    """
    if hasattr( uri, '__iter__' ):
        #test if it is iterable: works for lists and generators, but not for strings
        return uri
    else:
        try:
            # try if it is a URL and if we can open it
            f = requests.get( uri ).text.split( '\n' )
        except ValueError:
            # assume it is a file object
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
    return np.array( Y ).astype( int )


def store_matrix(matrix = '', output_dir_path = '', out_file_name = '', output_format = ''):
    if not os.path.exists( output_dir_path ) :
        os.mkdir( output_dir_path )
    full_out_file_name = os.path.join( output_dir_path, out_file_name )
    if output_format == "MatrixMarket":
        if len( matrix.shape ) == 1:
            raise Exception( "'MatrixMarket' format supports only 2D dimensional array and not vectors" )
        else:
            io.mmwrite( full_out_file_name, matrix, precision = None )
    elif output_format == "numpy":
        np.save( full_out_file_name, matrix )
    elif output_format == "joblib":
        joblib.dump( matrix, full_out_file_name )
    elif output_format == "text":
        with open( full_out_file_name, "w" ) as f:
            if len( matrix.shape ) == 1:
                for x in matrix:
                    f.write( "%s\n" % ( x ) )
                #data_str = map(str, matrix)
                #f.write('\n'.join(data_str))
            else:
                raise Exception( "Currently 'text' format supports only mono dimensional array and not matrices" )


def dump(obj, output_dir_path = '', out_file_name = ''):
    if not os.path.exists(output_dir_path) :
        os.mkdir(output_dir_path)
    full_out_file_name = os.path.join(output_dir_path, out_file_name) + ".pkl"
    joblib.dump(obj, full_out_file_name) 

def load(output_dir_path = '', out_file_name = ''):
    full_out_file_name = os.path.join(output_dir_path, out_file_name) + ".pkl"
    obj=joblib.load(full_out_file_name) 
    return obj