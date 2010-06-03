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

"""Vector Normalization
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__licence__ = 'GPL V3'

import hadoopy


class Mapper(object):
    def map(self, vecid, vector):
        """Parse vectors and emit using the order inversion design pattern.

        Args:
            vecid: vecid (int)
            vector: vector (iterator of numeric values)

        Yields:
            A tuple in the form of (key, value)
            key: (dim, flag) (tuple of (int, int))
            value: (vecid, value) (tuple of (int, float))
        """
        for dim, val in enumerate(vector):
            yield (dim, 0), (vecid, val)
            yield (dim, 1), (vecid, val)


class Reducer(object):
    def _update_extrema(self, v):
        self.m = float(min(self.m, v))
        self.M = float(max(self.M, v))

    def configure(self):
        self.m = self.M = self.p = None

    def reduce(self, key, tuples):
        """Outputs normalized values.

        Args:
            key: (dim, flag) (tuple of (int, int))
            tuples: (vecid, value) (tuple of (int, float))

        Yields:
            A tuple in the form of (key, value)
            key: vecid (int)
            value: (dim, value) (tuple of (int, int))
        """
        dim, flag = key
        if self.p != dim:
            self.m = float('inf')
            self.M = float('-inf')
            self.p = dim
        if flag == 0:
            for vecid, val in tuples:
                self._update_extrema(val)
        else:
            for vecid, val in tuples:
                val = (val - self.m) / (self.M - self.m)
                yield vecid, (dim, val)


if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
