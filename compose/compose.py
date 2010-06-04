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

"""Hadoopy Homography Composition
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__licence__ = 'GPL V3'

import os

import hadoopy
import numpy as np


class Mapper(object):
    def _extract_node_from(self, adjlist):
        return adjlist[0][0]

    def map(self, i, adjlist):
        """Emits each adjacency list with a flag if it belongs to the node.

        Args:
            i: node id
            adjlist: adjacency list in the form [(i, from0, H_i_0), ...]

        Yields:
            A tuple in the form of (key, value)
            key: node\tflag where flag is 0 if the adjlist belongs to the node
            value: adjlist
        """
        if self._extract_node_from(adjlist) == i:
            yield '%s\t%d' % (i, 0), adjlist
        else:
            yield '%s\t%d' % (i, 1), adjlist


class Reducer(object):
    def _first_iteration(self):
        try:
            return os.environ['IR_FIRST_ITER'] != 'False'
        except KeyError:
            return True

    def _find_to_h(self, to_find):
        for from_node, to_node, h_from_to in self.a:
            if to_node == to_find:
                return h_from_to

    def _update_adj_list(self, i, edge):
        from_node, to_node, h_from_to = edge[0:3]
        orig_to = set([x[1] for x in self.a_orig])
        cur_to = set([x[1] for x in self.a])
        if to_node not in orig_to:
            self.o.append(to_node)
        if to_node not in cur_to:
            # Let the current node (i in this case) be A, a node on it's adj
            # list is B (from_node in this case), and a node that is not in
            # A's adjacency list but is in B's be C (to_node in this case).
            # h_from_to is H_B_C and we need to find H_A_B from our adjacency
            # list because we want to end up with H_A_C.
            # We do this below as H_A_C = H_A_B * H_B_C
            h = np.dot(self._find_to_h(from_node), h_from_to)
            self.a.append((i, to_node, h))

    def configure(self):
        self.o = []
        self.a_orig = None
        self.a = None

    def reduce(self, key, adjlists):
        """Composes homographies to extend adjlist to new nodes.

        Args:
            key: node\tflag where flag is 0 if the adjlist belongs to the node
            adjlists: adjacency lists in the form [(i, from0, H_i_0), ...]

        Yields:
            A tuple in the form of (key, value)
            key: node id
            value: adjlist
        """
        i, g = map(int, key.split('\t', 1))
        if g == 0:
            for x in self.close():
                yield x
            self.a_orig = adjlists.next()
            self.a = list(self.a_orig)  # Make copy
            if self._first_iteration():
                for f, t, H in self.a_orig:
                    self.o.append(t)
            else:
                self.o.append(i)
        else:
            for adjlist in adjlists:
                for edge in adjlist:
                    self._update_adj_list(i, edge)

    def close(self):
        for t in self.o:
            yield t, self.a
        self.o = []

if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
