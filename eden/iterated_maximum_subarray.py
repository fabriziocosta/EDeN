def compute_maximum_subarray(score_vector = None):
    begin_temp = begin = end = 0
    start_val = score_vector[0]
    max_ending_here = max_so_far = start_val
    for pos,x in enumerate(score_vector[1:],1):
        if max_ending_here < 0:
            max_ending_here = x
            begin_temp = pos
        else:
            max_ending_here = max_ending_here + x
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            begin = begin_temp
            end = pos
    return begin,end


def compute_iterated_maximum_subarray(seq = None, score = None, min_subarray_size = None, max_subarray_size = None):
    subarray_list = []
    while 1 :
    	#find (begin,end) of subarray in each element
    	begin,end = compute_maximum_subarray(score_vector = score)
    	if end - begin < min_subarray_size -1 :
    		break
        else :
	    	#extract maximum subarray
	    	#subarray = seq[begin:end + 1]
            #NOTE: in order to account for border effects we select +1 element on the left and on the right
            first = max(0,begin - 1)
            last = min(len(seq),end + 1 + 1)
            subarray = seq[first : last]
            if max_subarray_size == -1 or len(subarray) <= max_subarray_size :
                #store data
                subarray_list += [subarray, begin, end, seq]
            #remove current subarray by zeoring importance values
            score[begin:end + 1] = [0.0] * len(subarray)
	    	#iterate
	return subarray_list


def compute_max_subarrays(graph = None, min_subarray_size = None, max_subarray_size = None ):
	#NOTE: the sequential order of nodes in the graph are used as the sequential constrain   
	#extract sequence of labels
	seq=[d['label'] for u,d in graph.nodes(data = True)]
	#extact sequence of scores
	score=[d['importance'] for u,d in graph.nodes(data = True)]
	#extract subarrays
	return compute_iterated_maximum_subarray(seq = seq, score = score, min_subarray_size = min_subarray_size, max_subarray_size = max_subarray_size)