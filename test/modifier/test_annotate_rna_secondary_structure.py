from eden.modifier.rna.structure_annotation import Annotator as Annotator
from eden.converter.rna import sequence_dotbracket_to_graph as fold


def test_annotator():
    # build a graph
    struct = "(((((((.((.......(((..((((.(((((((...........))))))).))))...))).)))))))))."
    seq = "GCGCCCGUAGCUCAAUUGGAUAGAGCGUUUGACUACGGAUCAAAAGGUUAGGGGUUCGACUCCUCUCGGGCGCG"
    g = fold(seq_info=seq, seq_struct=struct)
    g.graph['structure'] = struct

    # annotate
    f = Annotator()
    # transform
    g = f.transform([g], part_name='name', part_id='id').next()

    # check if annotation is ok
    names = set()
    ids = set()
    for n, d in g.nodes(data=True):
        assert len(d['name']) == 1
        assert len(d['id']) == 1
        names.add(d['name'][0])
        ids.add(d['id'][0])

    assert len(ids) == 11
    assert len(names) == 4
