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

"""Test BoF
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import unittest
import hadoopy
import os
import pickle
import numpy as np

from bof_maponly import Mapper


class Test(hadoopy.Test):
    def __init__(self, *args, **kw):
        super(Test, self).__init__(*args, **kw)

    def setUp(self):
        clusters = [np.array([4., 4.]),
                    np.array([6., 6.])]
        with open('clusters.pkl', 'w') as fp:
            pickle.dump(clusters, fp, 2)

    def tearDown(self):
        os.remove('clusters.pkl')

    def test_map(self):
        test_in = []
        test_in.append((0, [np.array([0., 0.]),
                            np.array([1., 1.]),
                            np.array([2., 2.]),
                            np.array([2., 3.]),
                            np.array([10., 10.])]))
        test_in.append((1, [np.array([10., 10.]),
                            np.array([11., 11.]),
                            np.array([12., 12.]),
                            np.array([12., 13.]),
                            np.array([0., 1.])]))
        test_in.append((2, [np.array([10., 10.]),
                            np.array([11., 11.]),
                            np.array([12., 12.]),
                            np.array([12., 13.])]))
        test_out = []
        test_out.append((0, {0: 4, 1: 1}))
        test_out.append((1, {0: 1, 1: 4}))
        test_out.append((2, {1: 4}))
        self.assertEqual(self.call_map(Mapper, test_in), test_out)

if __name__ == '__main__':
    unittest.main()
