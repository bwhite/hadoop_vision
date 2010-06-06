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

"""Test Sliding Window
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import unittest
import hadoopy

import Image
import numpy as np
from slidingwindow import Mapper, Reducer


class Test(hadoopy.Test):
    def __init__(self, *args, **kw):
        super(Test, self).__init__(*args, **kw)

    def test_mapreduce0(self):
        image = Image.open('test_data/test0.png')
        image_size = image.size
        image = np.fromstring(image.tostring(), dtype=np.uint8)
        image = image.reshape((image_size[1], image_size[0]))
        coords = [np.array([y, x, y + 1, x + 1])
                  for x in range(0, image_size[0])
                  for y in range(0, image_size[1])]
        test_in = [([0., 0.], (coords, image))]
        test_out = [([4.0, 4.0, 5.0, 5.0], 255.0)]
        kv = self.shuffle_kv(self.call_map(Mapper, test_in))
        result = self.call_reduce(Reducer, kv)
        self.assertEqual(result, test_out)

    def test_mapreduce1(self):
        image = Image.open('test_data/test1.png')
        image_size = image.size
        image = np.fromstring(image.tostring(), dtype=np.uint8)
        image = image.reshape((image_size[1], image_size[0]))
        coords = [np.array([y, x, y + 1, x + 1])
                  for x in range(0, image_size[0])
                  for y in range(0, image_size[1])]
        test_in = [([0., 0.], (coords, image))]
        test_out = [([0.0, 0.0, 1.0, 1.0], 255.0),
                    ([4.0, 4.0, 5.0, 5.0], 255.0),
                    ([9.0, 9.0, 10.0, 10.0], 255.0)]
        kv = self.shuffle_kv(self.call_map(Mapper, test_in))
        result = self.call_reduce(Reducer, kv)
        self.assertEqual(result, test_out)

if __name__ == '__main__':
    unittest.main()
