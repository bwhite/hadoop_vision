#!/usr/bin/env python
import cPickle as pickle
import numpy as np
import hadoopy


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

    def _extend_point(self, point):
        point = np.resize(point, len(point) + 1)
        point[-1] = 1
        return point

    def configure(self):
        self.clusters = self._load_clusters()

    def map(self, i, point):
        n = self._nearest_cluster_id(self.clusters, point)
        point = self._extend_point(point)
        yield n, point


class Reducer(object):
    def _compute_centroid(self, s):
        return s[0:-1] / s[-1]

    def reduce(self, n, points):
        s = 0
        for p in points:
            s += p
        m = self._compute_centroid(s)
        yield n, m


if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
