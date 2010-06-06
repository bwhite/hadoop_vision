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

"""Hadoopy Classifier Training
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import hadoopy
import numpy as np


class Mapper(object):
    def _parse_model_ids(self, metadata):
        """Extract the model_ids from the metadata

        Args:
            metadata: Dict of the form {'model_ids': [id0, ...], 'positive_id': id}

        Returns:
            The model_ids
        """
        return metadata['model_ids']

    def _parse_positive_id(self, metadata):
        """Extract the positive_id from the metadata

        Args:
            metadata: Dict of the form {'model_ids': [id0, ...], 'positive_id': id}

        Returns:
            The positive_id
        """
        return metadata['positive_id']

    def _compute_feature(self, image):
        """Compute an image feature (the intensity mean as a simple example)

        Args:
            image: Image as a 2d np array

        Returns:
            An image feature to be passed to the classifier
        """
        return np.mean(image)

    def map(self, metadata, image):
        """Perform a classification in a 'sliding window' pattern

        Args:
            metadata: Dict of the form {'model_ids': [id0, ...], 'positive_id': id}
            value: Image as a 2d np array

        Yields:
            A tuple in the form of (key, value)
            key: model_id
            value: A tuple in the form of (feature, polarity) as (np array, int)
                with polarity as -1 or 1
        """
        model_ids = self._parse_model_ids(metadata)
        positive_id = self._parse_positive_id(metadata)
        feature = self._compute_feature(image)
        for model_id in model_ids:
            if model_id == positive_id:
                yield model_id, (feature, 1)
            else:
                yield model_id, (feature, -1)


class Reducer(object):
    def _init_model(self):
        """Generate an initial model (in this case one that stores everything)

        Returns:
            A blank model
        """
        return {1: [], -1: []}

    def _update_model(self, feature, polarity, model):
        """Add a feature to the model

        Args:
            feature: An image feature (simply stored)
            polarity: -1 or 1 signifying the feature polarity
        """
        model[polarity].append(feature)

    def reduce(self, model_id, tuples):
        """Collect neighboring classifications and emit coord if local maxima

        Args:
            model_id: model_id
            tuples: Tuples in the form of (feature, polarity) as (np array, int)

        Yields:
            A tuple in the form of (key, value)
            key: model_id
            value: model resulting from the _update_model method
        """
        model = self._init_model()
        for feature, polarity in tuples:
            self._update_model(feature, polarity, model)
        yield model_id, model

if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
