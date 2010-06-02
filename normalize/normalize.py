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
    def map(self, i, V):
        """Parse vectors and emit using the order inversion design pattern.

        Args:
            i: vecid (int)
            V: vector (iterator of numeric values)

        Yields:
            A tuple in the form of (key, value)
            key: (dim, flag) (tuple of (int, int))
            value: (vecid, value) (tuple of (int, float))
        """
        for d, v in enumerate(V):
            yield (d, 0), (i, v)
            yield (d, 1), (i, v)


class Reducer(object):
    def _update_extrema(self, v):
        self.m = float(min(self.m, v))
        self.M = float(max(self.M, v))

    def configure(self):
        self.m = self.M = self.p = None

    def reduce(self, key, tuples):
        """

        Args:
            key: (dim, flag) (tuple of (int, int))
            tuples: (vecid, value) (tuple of (int, float))

        Yields:
            A tuple in the form of (key, value)
            key: vecid (int)
            value: (dim, value) (tuple of (int, int))
        """
        d, f = key
        if self.p != d:
            self.m = float('inf')
            self.M = float('-inf')
            self.p = d
        if f == 0:
            for i, v in tuples:
                self._update_extrema(v)
        else:
            for i, v in tuples:
                v = (v - self.m) / (self.M - self.m)
                yield i, (d, v)


if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
