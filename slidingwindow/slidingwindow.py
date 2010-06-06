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

"""Hadoopy Sliding Window
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import hadoopy
import numpy as np


class Mapper(object):
    def __init__(self):
        self._thresh = 1.
        r = 3  # Non-max suppression neighbor radius
        self._neighbor_multiples = [np.array([y, y + 1, x, x + 1])
                                    for x in range(-r, r + 1)
                                    for y in range(-r, r + 1)
                                    if not x == y == 0]

    def _classify(self, image, coord):
        """Dummy classification.  Confidence is the mean of the pixels

        Args:
            image: 2d image (np array)
            coord: ROI location (y_tl, x_tl, y_br, x_br) (np array)

        Returns:
            A float representing class confidence
        """
        return float(np.mean(image[coord[0]:coord[2], coord[1]:coord[3]]))

    def _neighbors(self, offset, coord):
        """Generates coordinates for neighboring ROIs

        Args:
            offset: pixel offset for the ROI (y, x) (np array)
            coord: ROI location (y_tl, x_tl, y_br, x_br) (np array)

        Returns:
            A generator of the neighbors (y_tl, x_tl, y_br, x_br) (np array)
        """
        height = coord[2] - coord[0]
        width = coord[3] - coord[1]
        scale = np.array([height, height, width, width])
        shift = np.array([offset[0], offset[0], offset[1], offset[1]])
        return (x * scale + shift for x in self._neighbor_multiples)

    def map(self, offset, value):
        """Perform a classification in a 'sliding window' pattern

        Args:
            offset: A numpy array representing the pixel offset for the ROI
            value: A tuple in the form (top left coords, image) as
                   (list of numpy arrays, 2d numpy array)

        Yields:
            A tuple in the form of (key, value)
            key: region coord (numpy array)
            value: A tuple in the form of (confidence, flag) as (float, int)
        """
        coords, image = value
        for coord in coords:
            p = self._classify(image, coord)
            if p > self._thresh:
                neighbor_coords = self._neighbors(offset, coord)
                for neighbor_coord in neighbor_coords:
                    yield neighbor_coord, (p, 0)
                yield coord + offset, (p, 1)


class Reducer(object):
    def reduce(self, coord, tuples):
        """Collect neighboring classifications and emit coord if local maxima

        Args:
            coord: ROI location (y_tl, x_tl, y_br, x_br) (np array)
            tuples: Tuples in the form of (confidence, flag) as (float, int)

        Yields:
            A tuple in the form of (key, value)
            key: ROI location (y_tl, x_tl, y_br, x_br) (np array)
            value: confidence (float)
        """
        max_flag = max_confidence = 0
        for confidence, flag in tuples:
            if max_confidence < confidence:
                max_confidence = confidence
                max_flag = flag
        if max_flag == 1:
            yield coord, max_confidence

if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
