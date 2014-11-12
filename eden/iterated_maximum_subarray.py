def find_smallest_positive(alist):
    #find first positive value
    minpos = -1
    for x in alist:
        if x > 0:
            minpos = x
            break
    if minpos > 0:
        #find smallest positive value 
        for x in alist:
            if x > 0 and x < minpos:
                minpos = x
    return minpos


def rebase_to_smallest_positive(alist):
    base = find_smallest_positive(alist)
    if base == -1:
        return None
    else:
        return [x - base for x in alist]


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
    original_score = score
    while 1 :
    	#find (begin,end) of subarray in each element
    	begin,end = compute_maximum_subarray(score_vector = score)
    	if end - begin < min_subarray_size -1 :
    		break
        else :
	    	#extract maximum subarray
            #NOTE: in order to account for border effects we select +1 element on the left 
            #and on the right
            first = max(0,begin - 1)
            last = min(len(seq),end + 1 + 1)
            subarray = seq[first : last]
            subarray_size = len(subarray)
            if max_subarray_size == -1 or subarray_size <= max_subarray_size :
                #store data
                acc = 0
                for x in original_score[begin : end + 1]:
                    acc += x
                subarray_list += [{'subarray':subarray, 'begin':first, 'end':last, 'size':subarray_size, 'seq' : seq, 'score':acc}]
            if subarray_size > max_subarray_size :
                #if the subarray is too large then rebase the score list, i.e. offset by the smallest positive value
                score = rebase_to_smallest_positive(score)
                if score is None:
                    break
            else :
                #remove current subarray by zeoring importance values
                score[first : last] = [0.0] * subarray_size
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