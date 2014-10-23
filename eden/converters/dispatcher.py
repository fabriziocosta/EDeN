def any_format_to_eden(input_file = "", format = "gspan", input_type = "file"):
	if format == "gspan":
		from eden.converters import gspan
		g_it = gspan.gspan_to_eden(input_file, input_type = input_type)
	elif format == "node_link_data":
		from eden.converters import node_link_data
		g_it = node_link_data.node_link_data_to_eden(input_file, input_type = input_type)
	elif format == "obabel":
		from eden.converters import obabel
		g_it = obabel.obabel_to_eden(input_file, input_type = input_type)
	elif format == "sequence":
		from eden.converters import sequence
		g_it = sequence.sequence_to_eden(input_file, input_type = input_type)
	else:
		raise Exception('Unknown format: %s' % format)
	return g_it