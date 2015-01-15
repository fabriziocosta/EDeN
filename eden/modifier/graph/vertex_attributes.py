import networkx as nx
import numpy as np 

def colorize(graph_list = None, labels = ['A','U','C','G']):
	values = np.linspace (0.0,1.0, num = len(labels))
	color_dict = dict(zip(labels,values))
	for g in graph_list:
		#iterate over nodes
		for n, d in g.nodes_iter(data = True):
			g.node[n]["level"] = color_dict.get(d['label'],0)
		yield g


def trapezoidal_reweighting(graph_list = None, high_weight = 1.0, low_weight = 0.1, high_weight_window_start = 0, high_weight_window_end = 1, low_weight_window_start = 0, low_weight_window_end = 1):
	"""
	Piece wise linear weight function between two levels with specified start end positions.
	
	high   ___
	low __/   \__

	"""
	#assert high_ weight > low_weight
	if high_weight < low_weight :
		raise Exception('high_weight (%f) must be higher than low_weight (%f)' % (high_weight, low_weight))

	#assert low_weight boundaries includes high_weight boundaries
	if high_weight_window_start > low_weight_window_end :
		raise Exception('high_weight_window_start (%d) must be lower than low_weight_window_end (%d)' % (high_weight_window_start, low_weight_window_end))
	if high_weight_window_start < low_weight_window_start :
		raise Exception('high_weight_window_start (%d) must be higher than low_weight_window_start (%d)' % (high_weight_window_start, low_weight_window_start))
	if high_weight_window_end < low_weight_window_start :
		raise Exception('high_weight_window_end (%d) must be higher than low_weight_window_start (%d)' % (high_weight_window_end, low_weight_window_start))
	if high_weight_window_end > low_weight_window_end :
		raise Exception('high_weight_window_end (%d) must be higher than low_weight_window_end (%d)' % (high_weight_window_end, low_weight_window_end))

	for g in graph_list:
		#iterate over nodes
		for n, d in g.nodes_iter(data = True):
			if 'position' not in d :
				#assert nodes must have position attribute
				raise Exception('Nodes must have "position" attribute') 
			#given the 'position' attribute of node assign weight according to piece wise linear weight function between two levels
			pos = d['position']
			if pos < low_weight_window_start:
				"""
				   ___
				__/   \__

				|
				"""
				g.node[n]["weight"] = low_weight
			elif pos >= low_weight_window_start and pos < high_weight_window_start:
				"""
				   ___
				__/   \__

				  |
				"""
				g.node[n]["weight"] = (high_weight - low_weight)/(high_weight_window_start - low_weight_window_start) * (pos - low_weight_window_start) + low_weight
			elif pos >= high_weight_window_start and pos < high_weight_window_end:
				"""
				   ___
				__/   \__

				    |
				"""
				g.node[n]["weight"] = high_weight
			elif pos >= high_weight_window_end and pos < low_weight_window_end:
				"""
				   ___
				__/   \__

				      |
				"""
				g.node[n]["weight"] = high_weight - (high_weight - low_weight)/(low_weight_window_end - high_weight_window_end) * (pos - high_weight_window_end)
			else:
				"""
				   ___
				__/   \__

				        |
				"""
				g.node[n]["weight"] = low_weight
		yield g