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

"""Test BG Sub
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__licence__ = 'GPL V3'

import gzip
import unittest
import pickle
import os

import hadoopy
import Image
import numpy as np

from bgsub import Mapper, Reducer

DATA_FN, SZ = 'pets2006-S1-T1-C-0-499.video_frame_data_small.pkl.gz', (45, 36)
#DATA_FN, SZ = 'pets2006-S1-T1-C-0-499.video_frame_data.pkl.gz', (720, 576)


class Test(hadoopy.Test):
    def __init__(self, *args, **kw):
        super(Test, self).__init__(*args, **kw)

    def test_mapreduce(self):
        test_in = pickle.load(gzip.open(data_fn))
        out = self.call_reduce(Reducer,
                               self.shuffle_kv(self.call_map(Mapper, test_in)))
        try:
            os.mkdir('output')
        except OSError:
            pass
        for image_id, image in out:
            image_fn = 'output/%s-%.10d.png' % image_id
            print(image_fn)
            image = image.replace('\x01', '\xff')
            Image.fromstring('L', image_size, image).save(image_fn)

if __name__ == '__main__':
    unittest.main()
