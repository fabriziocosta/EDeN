def solve_maximum_subarray_problem(score_vec):
    begin_temp = begin = end = 0
    start_val = score_vec[0]
    max_ending_here = max_so_far = start_val
    for pos,x in enumerate(score_vec[1:],1):
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


def compute_iterated_maximum_subarray(seq='', score='', min_motif_size=''):
    motif_list = []
    while 1:
    	#find (begin,end) of motif in each element
    	begin,end = solve_maximum_subarray_problem(score)
    	if end - begin < min_motif_size -1:
    		break
    	else :
	    	#extract maximum subarray
	    	#motif = seq[begin:end + 1]
	    	motif = seq[begin-1:end + 1 +1]
	    	#save motif, begin, end, seq, score_vec
	    	motif_item = [motif, begin, end, seq]
	    	motif_list += motif_item
	    	#remove current motif by zeoring importance values
	    	score[begin:end + 1] = [0.0] * len(motif)
	    	#iterate
	return motif_list


def extract_motifs(graph='', min_motif_size=''):
	#NOTE: the sequential order of nodes in the graph are used as the sequential constrain   
	#extract sequence of labels
	seq=[d['label'] for u,d in graph.nodes(data=True)]
	#extact sequence of scores
	score=[d['importance'] for u,d in graph.nodes(data=True)]
	#extract motifs
	return compute_iterated_maximum_subarray(seq = seq, score = score, min_motif_size = min_motif_size)