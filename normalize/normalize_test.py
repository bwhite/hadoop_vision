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

"""Test Vector Normalization
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__licence__ = 'GPL V3'

import unittest
import hadoopy
import os
from normalize import Mapper, Reducer


class Test(hadoopy.Test):
    def __init__(self, *args, **kw):
        super(Test, self).__init__(*args, **kw)

    def test_map(self):
        test_in = [(0, [9, 6]),
                   (1, [0, 1]),
                   (2, [1, 0]),
                   (3, (3, 6))]
        test_out = [((0, 0), (0, 9)),
                    ((0, 1), (0, 9)),
                    ((1, 0), (0, 6)),
                    ((1, 1), (0, 6)),
                    ((0, 0), (1, 0)),
                    ((0, 1), (1, 0)),
                    ((1, 0), (1, 1)),
                    ((1, 1), (1, 1)),
                    ((0, 0), (2, 1)),
                    ((0, 1), (2, 1)),
                    ((1, 0), (2, 0)),
                    ((1, 1), (2, 0)),
                    ((0, 0), (3, 3)),
                    ((0, 1), (3, 3)),
                    ((1, 0), (3, 6)),
                    ((1, 1), (3, 6))]
        self.assertEqual(self.call_map(Mapper, test_in), test_out)

    def test_reduce(self):
        test_in = [((0, 0), (0, 9)),
                   ((0, 1), (0, 9)),
                   ((1, 0), (0, 6)),
                   ((1, 1), (0, 6)),
                   ((0, 0), (1, 0)),
                   ((0, 1), (1, 0)),
                   ((1, 0), (1, 1)),
                   ((1, 1), (1, 1)),
                   ((0, 0), (2, 1)),
                   ((0, 1), (2, 1)),
                   ((1, 0), (2, 0)),
                   ((1, 1), (2, 0)),
                   ((0, 0), (3, 3)),
                   ((0, 1), (3, 3)),
                   ((1, 0), (3, 6)),
                   ((1, 1), (3, 6))]
        test_out = [(0, (0, 1.0)),
                    (1, (0, 0.0)),
                    (2, (0, 0.1111111111111111)),
                    (3, (0, 0.33333333333333331)),
                    (0, (1, 1.0)),
                    (1, (1, 0.16666666666666666)),
                    (2, (1, 0.0)),
                    (3, (1, 1.0))]
        self.assertEqual(self.call_reduce(Reducer,
                                          self.shuffle_kv(test_in)),
                         test_out)

if __name__ == '__main__':
    unittest.main()
