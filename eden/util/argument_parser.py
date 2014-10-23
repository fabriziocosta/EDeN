def setup_common_parameters(parser):
    """
    Common commandline arguments.
    """
    parser.add_argument("-i", "--input-file",
    	dest = "input_file",
    	help = "Path to your graph file.", 
    	required = True)
    parser.add_argument("-f", "--format",  choices = ["gspan", "node_link_data", "obabel", "sequence"],
    	help = "File format.", 
    	default = "gspan")
    parser.add_argument( "-r","--radius",
        type = int, 
        help = "Size of the largest radius used in EDeN.", 
        default = 2)
    parser.add_argument( "-d", "--distance",
        type = int, 
        help = "Size of the largest distance used in EDeN.", 
        default = 5)
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
		help = "Number of bits used to encode fetures, hence the feature space has size x \in IR 2^num-bits .", 
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