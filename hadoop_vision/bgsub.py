#!/usr/bin/env python
import os
import cPickle as pickle

import numpy as np

import hadoopy


class Mapper(object):

    @staticmethod
    def _compute_blockid(image_id):
        images_in_block = 300
        returun image_id / images_in_block
        
    def map(self, image_id, image):
        block_id = self._compute_blockid(image_id)
        yield (block_id, 1), (image_id, image)
        yield (block_id, 2), (image_id, image)

class Reducer(object):

    def reduce(self, key, values):
        block_id, flag = key
        if flag == 1:
            self._handle_flag1(values)
        else:
            self._handle_flag2(values)

    def _handle_flag1(self, values):
        for image_id, image in values:
            image = np.fromstring(image, dtype=np.float32)
            try:
                c += 1
                s += image
                ss += image * image
            except NameError:
                c = 1
                s = image
                ss = image * image
        self.m = s / c
        self.v = (ss - s * s / c) / c

    def _handle_flag2(self, values):
        for image_id, image in values:
            image = np.fromstring(image, dtype=np.float32)
            diff = image - self.m
            b = diff * diff < 6.25 * self.v
            yield image_id, b.tostring()


if __name__ == "__main__":
    if hadoopy.run(Mapper, reducer):
        hadoopy.print_doc_quit(__doc__)
