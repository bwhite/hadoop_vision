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

"""Hadoopy Bag-of-Features (BoF)
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import pickle

import hadoopy
import numpy as np


class Mapper(object):
    def _load_clusters(self):
        with open('clusters.pkl') as fp:
            return pickle.load(fp)

    def _nearest_cluster_id(self, clusters, point):
        """Find L2 squared nearest neighbor

        Args:
            clusters: A numpy array of shape (M, N). (N=Dims, M=NumClusters)
            point: A numpy array of shape (N,) or (1, N). (N=Dims)
        Returns:
            An int representing the nearest neighbor index into clusters.
        """
        dist = point - clusters
        dist = np.sum(dist * dist, 1)
        return int(np.argmin(dist))

    def configure(self):
        self.clusters = self._load_clusters()

    def map(self, imageid, feature):
        """

        Args:
            imageid: An ID that is directly passed to the output
            feature: A numpy array

        Yields:
            A tuple in the form of (key, value)
            key: imageid
            value: clusterid is the nearest cluster
        """
        clusterid = self._nearest_cluster_id(self.clusters, feature)
        yield imageid, clusterid

class Reducer(object):
    def _update_histogram(self, clusterid, histogram):
        try:
            histogram[clusterid] += 1
        except KeyError:
            histogram[clusterid] = 1

    def reduce(self, imageid, clusterids):
        """

        Args:
            imageid: An ID that is directly passed to the output
            clusterids: Each is an int

        Yields:
            A tuple in the form of (key, value)
            key: imageid
            value: histogram as a dict of (dim, val) (int, int)
        """
        histogram = {}
        for clusterid in clusterids:
            self._update_histogram(clusterid, histogram)
        yield imageid, histogram

if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
