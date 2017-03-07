from eden.converter.fasta import sequence_to_eden
import eden.modifier.graph.vertex_attributes as va


class TestEdenModifierGraphVertexAttributes:

    def test_symmetric_reweighting_no_annotation(self):
        """This function always expectes annotation of the center position to be
        set using format "center:int" in the fasta header."""
        graph = sequence_to_eden([("ID", "ACGUACGUAC")])
        try:
            graph = va.symmetric_trapezoidal_reweighting(graph,
                                                         high_weight=1,
                                                         low_weight=0,
                                                         radius_high=1,
                                                         distance_high2low=2)
            [x["weight"] for x in graph.next().node.values()]
        except AssertionError:
            pass
        else:
            raise Exception('ExpectedException not thrown')

    def test_symmetric_reweighting_noninteger_annotation(self):
        """This function always expectes annotation of the center position to be
        set using format "center:int" in the fasta header."""
        graph = sequence_to_eden([("ID center:not_an_integer", "ACGUACGUAC")])
        try:
            graph = va.symmetric_trapezoidal_reweighting(graph,
                                                         high_weight=1,
                                                         low_weight=0,
                                                         radius_high=1,
                                                         distance_high2low=2)
            [x["weight"] for x in graph.next().node.values()]
        except ValueError:
            pass
        else:
            raise Exception('ExpectedException not thrown')
