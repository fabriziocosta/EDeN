__author__ = 'jl'

import unittest
from eden.converter.molecule import obabel


class TestOBabel3d(unittest.TestCase):

    def setUp(self):
        self.converter = obabel.OBabelConverter()
        with open('../examples/data/tryptophan.smi') as inputfile:
            self.data = inputfile.readlines()
        # self.converter.obabel_to_eden3d(data)

    def test_find_nearest_neighbors(self):

        graph = self.converter.obabel_to_eden3d(input=self.data)
        print graph


if __name__ == '__main__':
    unittest.main()
