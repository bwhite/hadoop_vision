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

"""Test Classifier Training
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import unittest
import os
import glob

import hadoopy
import Image
import numpy as np
from classtrain import Mapper, Reducer


class Test(hadoopy.Test):
    def __init__(self, *args, **kw):
        super(Test, self).__init__(*args, **kw)
    
    def _load_image_as_np(self, fn):
        image = Image.open(fn).convert('L')
        image_size = image.size
        image = np.fromstring(image.tostring(), dtype=np.uint8)
        return image.reshape((image_size[1], image_size[0]))

    def test_mapreduce(self):
        # Test data is white cat, black dog, and neither
        test_in = []
        all_data = {'cat': [], 'dog': [], 'misc': []}
        for fn in sorted(glob.glob('test_data/cat*.jpg')):
            image = self._load_image_as_np(fn)
            all_data['cat'].append(image)
        for fn in sorted(glob.glob('test_data/dog*.jpg')):
            image = self._load_image_as_np(fn)
            all_data['dog'].append(image)
        for fn in sorted(glob.glob('test_data/misc*.jpg')):
            image = self._load_image_as_np(fn)
            all_data['misc'].append(image)
        model_ids = ['cat', 'dog']
        # Note that a model is NOT made for misc as it isn't in model_ids
        metadatas = {'dog': {'model_ids': model_ids, 'positive_id': 'dog'},
                     'cat': {'model_ids': model_ids, 'positive_id': 'cat'},
                     'misc': {'model_ids': model_ids, 'positive_id': 'misc'}}
        for obj_class, images in all_data.items():
            metadata = metadatas[obj_class]
            for image in images:
                test_in.append((metadata, image))
        test_out = [('cat', {1: [175.46912202380952, 205.85798313372547,
                                 104.77855072463768],
                             -1: [95.844938271604931, 105.45872333895964,
                                  77.342401960784315, 100.9815188172043,
                                  94.968019323671498, 123.61050061050061]}),
                    ('dog', {1: [100.9815188172043, 94.968019323671498,
                                 123.61050061050061],
                             -1: [95.844938271604931, 105.45872333895964,
                                  77.342401960784315, 175.46912202380952,
                                  205.85798313372547, 104.77855072463768]})]
        result = self.call_reduce(Reducer,
                                  self.shuffle_kv(self.call_map(Mapper, test_in)))
        self.assertEqual(result, test_out)

if __name__ == '__main__':
    unittest.main()
