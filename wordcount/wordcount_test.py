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

"""Test Word Count
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import unittest
import hadoopy

from wordcount import Mapper, Reducer


class Test(hadoopy.Test):
    def __init__(self, *args, **kw):
        super(Test, self).__init__(*args, **kw)

    def test_map(self):
        test_in = [(0, 'a b c'),
                   (1, 'c b a'),
                   (2, ''),
                   (3, 'd d c')]
        test_out = [('a', 1),
                    ('b', 1),
                    ('c', 1),
                    ('c', 1),
                    ('b', 1),
                    ('a', 1),
                    ('d', 1),
                    ('d', 1),
                    ('c', 1)]
        self.assertEqual(self.call_map(Mapper, test_in), test_out)

    def test_reduce(self):
        test_in = [('a', 1),
                   ('b', 1),
                   ('c', 1),
                   ('c', 1),
                   ('b', 1),
                   ('a', 1),
                   ('d', 1),
                   ('d', 1),
                   ('c', 1)]
        test_out = [('a', 2),
                    ('b', 2),
                    ('c', 3),
                    ('d', 2)]
        self.assertEqual(self.call_reduce(Reducer,
                                          self.shuffle_kv(test_in)),
                         test_out)

if __name__ == '__main__':
    unittest.main()
