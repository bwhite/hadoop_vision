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
__license__ = 'GPL V3'

import unittest
import hadoopy
import os

import numpy as np

from compose import Mapper, Reducer


def make_chain(s, l):
    test_in = []
    e = np.eye(3)
    for x in range(s, l - 1):
        if x == s:
            cur = (s, [(s, s, e), (s, s + 1, e)])
        else:
            cur = (x, [(x, x - 1, e), (x, x, e), (x, x + 1, e)])
        test_in.append(cur)
    cur = (l - 1, [(l - 1, l - 2, e), (l - 1, l - 1, e)])
    test_in.append(cur)
    return test_in


class TestImageReg(hadoopy.Test):
    def __init__(self, *args, **kw):
        super(TestImageReg, self).__init__(*args, **kw)

    def reset_first(self):
        try:
            del os.environ['IR_FIRST_ITER']
        except KeyError:
            pass

    def rout_tolist(self, s):
        return [(x[0], [(a, b, c.tolist()) for a, b, c in sorted(x[1])])
                for x in sorted(s)]

    def test_map(self):
        self.reset_first()
        e = np.eye(3)
        test_in = [(0, [(0, 0, e), (0, 1, e)]),
                   (1, [(1, 0, e), (1, 1, e), (1, 2, e)]),
                   (2, [(2, 1, e), (2, 2, e), (2, 3, e)]),
                   (3, [(3, 2, e), (3, 3, e)])]
        test_out = [('0\t0', [(0, 0, e), (0, 1, e)]),
                    ('1\t0', [(1, 0, e), (1, 1, e), (1, 2, e)]),
                    ('2\t0', [(2, 1, e), (2, 2, e), (2, 3, e)]),
                    ('3\t0', [(3, 2, e), (3, 3, e)])]
        self.assertEqual(self.call_map(Mapper, test_in), test_out)

    def test_reduce_shift(self):
        os.environ['IR_FIRST_ITER'] = 'False'
        e = np.eye(3)
        s = np.mat('[1,0,5;0,1,5;0,0,1.]')
        s_inv = s.I.A
        s = s.A
        ss = np.dot(s, s)
        ss_inv = np.dot(s_inv, s_inv)
        test_in = [('0\t0', [(0, 0, e), (0, 1, s)]),
                   ('1\t1', [(0, 0, e), (0, 1, s)]),
                   ('1\t0', [(1, 0, s_inv), (1, 1, e), (1, 2, s)]),
                   ('0\t1', [(1, 0, s_inv), (1, 1, e), (1, 2, s)]),
                   ('2\t1', [(1, 0, s_inv), (1, 1, e), (1, 2, s)]),
                   ('2\t0', [(2, 1, s_inv), (2, 2, e)]),
                   ('1\t1', [(2, 1, s_inv), (2, 2, e)])]
        test_out = [(0, [(0, 0, e), (0, 1, s), (0, 2, ss)]),
                    (2, [(0, 0, e), (0, 1, s), (0, 2, ss)]),
                    (1, [(1, 0, s_inv), (1, 1, e), (1, 2, s)]),
                    (2, [(2, 0, ss_inv), (2, 1, s_inv), (2, 2, e)]),
                    (0, [(2, 0, ss_inv), (2, 1, s_inv), (2, 2, e)])]
        kv = self.rout_tolist(self.call_reduce(Reducer, self.shuffle_kv(test_in)))
        self.assertEqual(kv, self.rout_tolist(test_out))

    def test_map1(self):
        self.reset_first()
        e = np.eye(3)
        test_in = [(0, [(0, 0, e), (0, 1, e)]),
                   (1, [(0, 0, e), (0, 1, e)]),
                   (0, [(1, 0, e), (1, 1, e), (1, 2, e)]),
                   (1, [(1, 0, e), (1, 1, e), (1, 2, e)]),
                   (2, [(1, 0, e), (1, 1, e), (1, 2, e)]),
                   (1, [(2, 1, e), (2, 2, e), (2, 3, e)]),
                   (2, [(2, 1, e), (2, 2, e), (2, 3, e)]),
                   (3, [(2, 1, e), (2, 2, e), (2, 3, e)]),
                   (2, [(3, 2, e), (3, 3, e)]),
                   (3, [(3, 2, e), (3, 3, e)])]
        test_out = [('0\t0', [(0, 0, e), (0, 1, e)]),
                    ('1\t1', [(0, 0, e), (0, 1, e)]),
                    ('0\t1', [(1, 0, e), (1, 1, e), (1, 2, e)]),
                    ('1\t0', [(1, 0, e), (1, 1, e), (1, 2, e)]),
                    ('2\t1', [(1, 0, e), (1, 1, e), (1, 2, e)]),
                    ('1\t1', [(2, 1, e), (2, 2, e), (2, 3, e)]),
                    ('2\t0', [(2, 1, e), (2, 2, e), (2, 3, e)]),
                    ('3\t1', [(2, 1, e), (2, 2, e), (2, 3, e)]),
                    ('2\t1', [(3, 2, e), (3, 3, e)]),
                    ('3\t0', [(3, 2, e), (3, 3, e)])]
        self.assertEqual(self.call_map(Mapper, test_in), test_out)

    def test_reduce0(self):
        self.reset_first()
        e = np.eye(3)
        test_in = [('0\t0', [(0, 0, e), (0, 1, e)]),
                    ('1\t0', [(1, 0, e), (1, 1, e), (1, 2, e)]),
                    ('2\t0', [(2, 1, e), (2, 2, e), (2, 3, e)]),
                    ('3\t0', [(3, 2, e), (3, 3, e)])]
        test_out = [(0, [(0, 0, e), (0, 1, e)]),
                    (1, [(0, 0, e), (0, 1, e)]),
                    (0, [(1, 0, e), (1, 1, e), (1, 2, e)]),
                    (1, [(1, 0, e), (1, 1, e), (1, 2, e)]),
                    (2, [(1, 0, e), (1, 1, e), (1, 2, e)]),
                    (1, [(2, 1, e), (2, 2, e), (2, 3, e)]),
                    (2, [(2, 1, e), (2, 2, e), (2, 3, e)]),
                    (3, [(2, 1, e), (2, 2, e), (2, 3, e)]),
                    (2, [(3, 2, e), (3, 3, e)]),
                    (3, [(3, 2, e), (3, 3, e)])]
        self.assertEqual(self.call_reduce(Reducer, self.groupby_kv(test_in)),
                         test_out)

    def test_reduce1(self):
        self.reset_first()
        e = np.eye(3)
        test_in = [('0\t0', [(0, 0, e), (0, 1, e)]),
                   ('0\t1', [(1, 0, e), (1, 1, e), (1, 2, e)]),
                   ('1\t0', [(1, 0, e), (1, 1, e), (1, 2, e)]),
                   ('1\t1', [(0, 0, e), (0, 1, e)]),
                   ('1\t1', [(2, 1, e), (2, 2, e), (2, 3, e)]),
                   ('2\t0', [(2, 1, e), (2, 2, e), (2, 3, e)]),
                   ('2\t1', [(1, 0, e), (1, 1, e), (1, 2, e)]),
                   ('2\t1', [(3, 2, e), (3, 3, e)]),
                   ('3\t0', [(3, 2, e), (3, 3, e)]),
                   ('3\t1', [(2, 1, e), (2, 2, e), (2, 3, e)])]
        test_out = [(0, [(0, 0, e), (0, 1, e), (0, 2, e)]),
                    (0, [(1, 0, e), (1, 1, e), (1, 2, e), (1, 3, e)]),
                    (0, [(2, 0, e), (2, 1, e), (2, 2, e), (2, 3, e)]),
                    (1, [(0, 0, e), (0, 1, e), (0, 2, e)]),
                    (1, [(1, 0, e), (1, 1, e), (1, 2, e), (1, 3, e)]),
                    (1, [(2, 0, e), (2, 1, e), (2, 2, e), (2, 3, e)]),
                    (1, [(3, 1, e), (3, 2, e), (3, 3, e)]),
                    (2, [(0, 0, e), (0, 1, e), (0, 2, e)]),
                    (2, [(1, 0, e), (1, 1, e), (1, 2, e), (1, 3, e)]),
                    (2, [(2, 0, e), (2, 1, e), (2, 2, e), (2, 3, e)]),
                    (2, [(3, 1, e), (3, 2, e), (3, 3, e)]),
                    (3, [(1, 0, e), (1, 1, e), (1, 2, e), (1, 3, e)]),
                    (3, [(2, 0, e), (2, 1, e), (2, 2, e), (2, 3, e)]),
                    (3, [(3, 1, e), (3, 2, e), (3, 3, e)])]
        kv = self.rout_tolist(self.call_reduce(Reducer, self.groupby_kv(test_in)))
        self.assertEqual(kv, self.rout_tolist(test_out))

    def _chain_iter(self, l):
        test_in = make_chain(0, l)
        #print(test_in)
        self.reset_first()
        i = 0
        full_set = set(range(l))
        while 1:
            kv = self.groupby_kv(self.sort_kv(self.call_map(Mapper, test_in)))
            test_in = self.call_reduce(Reducer, kv)
            os.environ['IR_FIRST_ITER'] = 'False'
            if i != 0 and len(test_in) == l:
                for test_edge in test_in:
                    test_full_set = set([x[1] for x in test_edge[1]])
                    self.assertEqual(test_full_set, full_set)
                break
            i += 1

    def test_chain(self):
        for i in range(1, 5):
            self._chain_iter(2 ** i)


if __name__ == '__main__':
    unittest.main()
