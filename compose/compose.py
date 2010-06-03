#!/usr/bin/env python
import os

import hadoopy
import numpy as np

class Mapper(object):
    def _extract_node_from(self, adjlist):
        return adjlist[0][0]

    def map(self, i, adjlist):
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
        i, g = map(int, key.split('\t', 1))
        if g == 0:
            for x in self.close():
                yield x
            self.a_orig = adjlists.next()
            self.a = list(self.a_orig) # Make copy
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
