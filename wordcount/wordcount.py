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

"""Hadoopy Wordcount Demo
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import hadoopy


class Mapper(object):
    def map(self, unused_docid, doc):
        """Take in a byte offset and a document, emit terms with count of 1.

        Args:
            unused_docid: byte offset (unused)
            doc: document as a string of terms delimited by whitespace

        Yields:
            A tuple in the form of (key, value)
            key: term (string)
            value: partial count (int)
        """
        for term in doc.split():
            yield term, 1


class Reducer(object):
    def reduce(self, term, counts):
        """Take in an iterator of counts for a word, sum them, and return sum.

        Args:
            term: word (string)
            counts: counts (int)

        Yields:
            A tuple in the form of (key, value)
            key: term (string)
            value: count (int)
        """
        _sum = 0
        for count in counts:
            _sum += count
        yield term, _sum


if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
