#!/usr/bin/env python
import os
import hadoopy

class Mapper(object):
    def _extract_node_from(self, adjlist):
        return adjlist[0][0]

    def map(self, i, adjlist):
        if self._extract_node_from(adjlist) == i:
            yield '%s\t%d' % (i, 0), adjlist
        else:
            yield '%s\t%d' % (i, 1), adjlist

class Reducer(object):
    def configure(self):
        self.o = []
        self.a_orig = None
        self.a = None

    def _first_iteration(self):
        try:
            return os.environ['IR_FIRST_ITER'] != 'False'
        except KeyError:
            return True

    def _update_adj_list(self, i, edge):
        from_node, to_node = edge[0:2]
        orig_to = set([x[1] for x in self.a_orig])
        cur_to = set([x[1] for x in self.a])
        if to_node not in orig_to:
            self.o.append(to_node)
        if to_node not in cur_to:
            self.a.append((i, to_node))

    def _handle_flag0(self, i, adjlists):
        for x in self.close():
            yield x
        self.a_orig = adjlists.next()
        self.a = list(self.a_orig) # Make copy
        if self._first_iteration():
            for f, t in self.a_orig:
                self.o.append(t)
        else:
            self.o.append(i)

    def _handle_flag1(self, i, adjlists):
        for l in adjlists:
            for e in l:
                self._update_adj_list(i, e)

    def reduce(self, key, values):
        i = int(key.split('\t', 1)[0])
        return self._handle_flag0(i, values) if key[-1] == '0' else self._handle_flag1(i, values)

    def close(self):
        for t in self.o:
            yield t, self.a
        self.o = []

if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
