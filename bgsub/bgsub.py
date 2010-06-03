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

"""Hadoopy Single Gaussian BG Sub
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__licence__ = 'GPL V3'

import hadoopy
import numpy as np


class Mapper(object):
    def _compute_blockid(self, image_id):
        images_in_block = 500
        return str(image_id[1] / images_in_block)

    def map(self, image_id, image):
        """Emits image using order inversion

        Args:
            image_id: A tuple in the form (video_id (str), frame_num (int))
            image: A uint8 bytestring representing the image

        Yields:
            A tuple in the form of (key, value)
            key: block_id-flag (string) where we partition on the block_id
            value: A tuple in the form of (image_id, image)
        """
        block_id = self._compute_blockid(image_id)
        yield block_id + '-0', (image_id, image)
        yield block_id + '-1', (image_id, image)


class Reducer(object):
    def _load_image(self, image):
        image = np.fromstring(image, dtype=np.uint8)
        return np.array(image, dtype=np.uint32)

    def reduce(self, key, values):
        """Emite bgsub masks for each image

        Args:
            key: block_id-flag (string) where we partition on the block_id
            values: Tuples in the form of (image_id, image)

        Yields:
            A tuple in the form of (key, value)
            key: image_id (string)
            value: bgsub mask as a uint8 bytestring with BG=0 and FG=1
        """
        values = ((d, self._load_image(i)) for d, i in values)
        if key[-1] == '0':
            c = s = ss = 0
            for d, i in values:
                c += 1
                s += i
                ss += i ** 2
            self.m = s / c
            self.v = (ss - s ** 2 / c) / c
        else:
            for d, i in values:
                b = (i - self.m) ** 2 > 6.25 * self.v
                yield d, b.tostring()

if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
