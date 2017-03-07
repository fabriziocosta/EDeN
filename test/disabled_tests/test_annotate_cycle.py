from eden.modifier.graph.cycle import Annotator as Annotator
import networkx as nx


def test_annotator():
    # making graph
    graph = nx.cycle_graph(5)
    graph.add_edge(3, 6)
    for n, d in graph.nodes(data=True):
        d['label'] = 'X'

    # annotate
    a = Annotator()
    graph = a.transform([graph], part_name='name', part_id='id').next()

    # test annotation
    cyc = 0
    other = 0
    ids = set()
    for n, d in graph.nodes(data=True):
        ids.add(d['id'][0])  # unpack because list unhashable
        if d['name'] == ['XXXXX']:
            cyc += 1
        if d['name'] == ['X']:
            other += 1

    assert cyc == 5
    assert other == 1
    assert len(ids) == 2
