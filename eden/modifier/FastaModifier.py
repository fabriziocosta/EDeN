import random

class Modifier(object):
    input = None
    input_type = None
    def __init__(self, input = None, input_type = None):
        """
        Takes a FASTA file yields a normalised FASTA file.

        Parameters
        ----------
        input : string
            A pointer to the data source.

        input_type : ['url','file','string_file']
            If type is 'url' then 'input' is interpreted as a URL pointing to a file.
            If type is 'file' then 'input' is interpreted as a file name.
            If type is 'list' then 'input' is interpreted as a list of strings. This is the default.
        """
        self.input = input
        if input_type is None:
            input_type = 'list'
        self.input_type = input_type

        #here call apply
        #pass modify as an argument function 

    def apply(self, **options):
        lines = self._to_FASTA()
        for line in lines:
            header_in = line
            seq_in = lines.next()
            seqs = self.modify(header = header_in, seq = seq_in, **options)
            for seq in seqs:
                yield seq

    
    def modify(self, header = None, seq = None, **options):
        yield header
        yield seq


    def _to_FASTA(self):
        if self.input_type == 'file':
            f = open(self.input)
        elif self.input_type == 'url':
            import requests
            f = requests.get(self.input).text.split('\n')
        elif self.input_type == 'list':
            f = self.input
        else:
            raise Exception('Unknown type: %s' % self.input_type)

        seq = ''
        for line in f:
            _line = line.strip()
            if _line:
                if _line[0] == '>':
                    #extract string from header
                    header = _line 
                    if len(seq) > 0:
                        yield prev_header
                        yield seq
                    seq = ''
                    prev_header = header
                else:
                    seq += _line
        if len(seq) > 0:
            yield prev_header
            yield seq


class one_line(Modifier):
    def modify(self, header = None, seq = None, **options):
        header_only = options.get('header_only',False)
        one_line = options.get('one_line',True)
        if one_line:
            yield  header + '\t' + seq
        if header_only:
            yield header
        

class insert_landmark(Modifier):
    def modify(self, header = None, seq = None, **options):
        landmark_relative_position = options.get('landmark_relative_position',0.5)
        landmark_char =  options.get('landmark_char','@')
        pos = int( len(seq) * landmark_relative_position )
        seq_out = seq[:pos] + landmark_char + seq[pos:]
        yield header
        yield seq_out


class shuffle(Modifier):
    def modify(self, header = None, seq = None, **options):
        times =  options.get('times',1)
        for i in range(times):
            seq_mod = [c for c in seq]
            random.shuffle(seq_mod)
            seq_out = ''.join(seq_mod)        
            yield header
            yield seq_out


class remove(Modifier):
    def modify(self, header = None, seq = None, **options):
        remove_char =  options.get('remove_char','-')
        if not remove_char in seq:
            yield header
            yield seq
            

class keep(Modifier):
    def modify(self, header = None, seq = None, **options):
        keep_char_list =  options.get('keep_char_list',['A','C','G','T','U'])
        skip = False
        for c in seq:
            if not c in keep_char_list:
                skip = True
                break
        if not skip:
            yield header
            yield seq


class split(Modifier):
    def modify(self, header = None, seq = None, **options):
        step =  options.get('step',10)
        window =  options.get('window',100)
        seq_len = len(seq)
        for start in range(0, seq_len, step):
            seq_out = seq[start : start + window]
            yield '%s START: %0.5d WINDOW: %0.5d' % (header, start, window)
            yield seq_out
