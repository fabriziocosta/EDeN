def FASTA_to_FASTA(input = None, input_type = None, **options):
    """
    Takes a FASTA file yields a normalised FASTA file.

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
    return _FASTA_to_FASTA(f, **options)      


def _FASTA_to_FASTA(data_str_list, **options):
    header_only = options.get('header_only',False)
    line_buffer = ''
    for line in data_str_list:
        _line = line.strip()
        if _line:
            if _line[0] == '>':
                #extract string from header
                header_str = _line[1:] 
                if len(line_buffer) > 0:
                    yield '>' + prev_header_str
                    if header_only == False:
                        yield line_buffer
                line_buffer = ''
                prev_header_str = header_str
            else:
                line_buffer += _line
    if len(line_buffer) > 0:
        yield '>' + prev_header_str
        if header_only == False:
            yield line_buffer