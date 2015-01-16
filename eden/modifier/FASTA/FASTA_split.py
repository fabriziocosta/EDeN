import random

def FASTA_split_to_FASTA(input = None, input_type = None, options = dict()):
    """
    Takes a FASTA file yields a FASTA file with split sequences.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    input_type : ['url','file','string_file']
        If type is 'url' then 'input' is interpreted as a URL pointing to a file.
        If type is 'file' then 'input' is interpreted as a file name.
        If type is 'list' then 'input' is interpreted as a list of strings.
    """
    input_types = ['url','file','list']
    assert(input_type in input_types),'ERROR: input_type must be one of %s ' % input_types

    if input_type == 'file':
        f = open(input,'r')
    elif input_type == 'url':
        import requests
        f = requests.get(input).text.split('\n')
    elif input_type == "list":
        f = input
    return _FASTA_split_to_FASTA(f, options = options)        


def _FASTA_split_to_FASTA(data_str_list, options = None):
    defaults = {'window':70, 'step':20, 'header_only':False}
    defaults.update(options)

    line_buffer = ''
    for line in data_str_list:
        _line = line.strip().upper()
        if _line:
            if _line[0] == '>':
                #extract string from header
                header_str = _line[1:] 
                seq_len = len(line_buffer)
                if seq_len > 0:
                    #split sequence in windows of size defaults['window'] starting in steps with increment defaults['step']
                    for start in range(0, seq_len, defaults['step']):
                        yield '>%s START: %d WINDOW: %d' % (prev_header_str, start, defaults['window'])
                        if defaults['header_only'] == False :
                            subseq = line_buffer[start : start + defaults['window']]
                            yield subseq
                line_buffer = ''
                prev_header_str = header_str
            else:
                line_buffer += _line
    seq_len = len(line_buffer)
    if seq_len > 0:
        #split sequence in windows of size defaults['window'] starting in steps with increment defaults['step']
        for start in range(0, seq_len, defaults['step']):
            yield '>%s START: %d WINDOW: %d' % (prev_header_str, start, defaults['window'])
            if defaults['header_only'] == False :
                subseq = line_buffer[start : start + defaults['window']]
                yield subseq