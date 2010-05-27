#!/usr/bin/env python
import unittest
import hadoopy
from image_reg import Mapper, Reducer

class TestImageReg(hadoopy.Test):
    
    def __init__(self, *args, **kw):
        super(TestImageReg, self).__init__(*args, **kw)

    def test_map(self):
        test_in = [(0, [(0, 0), (0, 1)]),
                   (1, [(1, 0), (1, 1), (1, 2)]),
                   (2, [(2, 1), (2, 2), (2, 3)]),
                   (3, [(3, 2), (3, 3)])]
        test_out = [('0\t0', [(0, 0), (0, 1)]),
                    ('1\t0', [(1, 0), (1, 1), (1, 2)]),
                    ('2\t0', [(2, 1), (2, 2), (2, 3)]),
                    ('3\t0', [(3, 2), (3, 3)])]
        self.assertEqual(self.call_map(Mapper, test_in), test_out)

    def test_reduce(self):
        test_in = [('0\t0', iter([[(0, 0), (0, 1)]])),
                   ('1\t0', iter([[(1, 0), (1, 1), (1, 2)]])),
                   ('2\t0', iter([[(2, 1), (2, 2), (2, 3)]])),
                   ('3\t0', iter([[(3, 2), (3, 3)]]))]
        test_out = [(0, [(0, 0), (0, 1)]),
                    (1, [(0, 0), (0, 1)]),
                    (0, [(1, 0), (1, 1), (1, 2)]),
                    (1, [(1, 0), (1, 1), (1, 2)]),
                    (2, [(1, 0), (1, 1), (1, 2)]),
                    (1, [(2, 1), (2, 2), (2, 3)]),
                    (2, [(2, 1), (2, 2), (2, 3)]),
                    (3, [(2, 1), (2, 2), (2, 3)]),
                    (2, [(3, 2), (3, 3)]),
                    (3, [(3, 2), (3, 3)])]
        self.assertEqual(self.call_reduce(Reducer, test_in), test_out)

if __name__ == '__main__':
    unittest.main()
