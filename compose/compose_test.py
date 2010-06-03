#!/usr/bin/env python
import unittest
import hadoopy
import os

import numpy as np

from compose import Mapper, Reducer

def make_chain(s, l):
    cnt = s
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

    def test_reduce(self):
        self.reset_first()
        ite = lambda x: iter([x])
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
        self.assertEqual(self.call_reduce(Reducer, self.groupby_kv(test_in)), test_out)

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

    def test_reduce1(self):
        self.reset_first()
        ite = lambda x: iter([x])
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
                    (1, [(0, 0, e), (0, 1, e), (0, 2, e)]),
                    (1, [(0, 0, e), (0, 1, e), (0, 2, e)]),
                    (0, [(1, 0, e), (1, 1, e), (1, 2, e), (1, 3, e)]),
                    (1, [(1, 0, e), (1, 1, e), (1, 2, e), (1, 3, e)]),
                    (2, [(1, 0, e), (1, 1, e), (1, 2, e), (1, 3, e)]),
                    (2, [(1, 0, e), (1, 1, e), (1, 2, e), (1, 3, e)]),
                    (1, [(2, 1, e), (2, 2, e), (2, 3, e), (2, 0, e)]),
                    (2, [(2, 1, e), (2, 2, e), (2, 3, e), (2, 0, e)]),
                    (3, [(2, 1, e), (2, 2, e), (2, 3, e), (2, 0, e)]),
                    (1, [(2, 1, e), (2, 2, e), (2, 3, e), (2, 0, e)]),
                    (2, [(3, 2, e), (3, 3, e), (3, 1, e)]),
                    (3, [(3, 2, e), (3, 3, e), (3, 1, e)]),
                    (2, [(3, 2, e), (3, 3, e), (3, 1, e)])]

    def _chain_iter(self, l):
        test_in = make_chain(0, l)
        #print(test_in)
        self.reset_first()
        i = 0
        full_set = set(range(l))
        while 1:
            test_in = self.call_reduce(Reducer, self.groupby_kv(self.sort_kv(self.call_map(Mapper, test_in))))
            os.environ['IR_FIRST_ITER'] = 'False'
            if i != 0 and len(test_in) == l:
                for test_edge in test_in:
                    self.assertEqual(set([x[1] for x in test_edge[1]]), full_set)
                break
            i += 1

    def test_chain(self):
        for i in range(1, 5):
            self._chain_iter(2**i)


if __name__ == '__main__':
    unittest.main()
