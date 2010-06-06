#!/usr/bin/env python
# (C) Copyright 2010 Brandyn A. White
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Test K-Means Clustering
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import unittest
import hadoopy
import os
import pickle
import numpy as np
from kmeans_imc import Mapper, Reducer


class Test(hadoopy.Test):
    def __init__(self, *args, **kw):
        super(Test, self).__init__(*args, **kw)
        self.Mapper = Mapper
        self.Reducer = Reducer

    def setUp(self):
        clusters = [np.array([4., 4.]),
                    np.array([6., 6.])]
        with open('clusters.pkl', 'w') as fp:
            pickle.dump(clusters, fp, 2)

    def tearDown(self):
        os.remove('clusters.pkl')

    def test_map(self):
        test_in = [(0, np.array([2., 2.])),
                   (1, np.array([1., 1.])),
                   (2, np.array([-1., -1.])),
                   (3, np.array([-2., -2.])),
                   (4, np.array([12., 12.])),
                   (5, np.array([11., 11.])),
                   (6, np.array([9., 9.])),
                   (7, np.array([8., 8.]))]
        test_out = [(0, np.array([0., 0., 4.])),
                    (1, np.array([40., 40., 4.]))]

        def tolist(s):
            return [(x[0], x[1].tolist()) for x in s]
        self.assertEqual(tolist(self.call_map(self.Mapper, test_in)),
                         tolist(test_out))

    def test_reduce(self):
        test_in = [(0, np.array([0., 0., 4.])),
                   (1, np.array([40., 40., 4.]))]
        test_out = [(0, np.array([0., 0.])),
                    (1, np.array([10., 10.]))]

        def tolist(s):
            return [(x[0], x[1].tolist()) for x in s]
        self.assertEqual(tolist(self.call_reduce(self.Reducer,
                                                 self.shuffle_kv(test_in))),
                         tolist(test_out))

if __name__ == '__main__':
    unittest.main()
