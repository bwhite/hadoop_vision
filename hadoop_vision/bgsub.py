#!/usr/bin/env python
import base64

import numpy as np
import StringIO
import Image

import hadoopy


class Mapper(object):

    @staticmethod
    def _compute_blockid(image_id):
        images_in_block = 500
        return str(int(image_id) / images_in_block)
        
    def map(self, video_id, value):
        image_id, image = value
        block_id = self._compute_blockid(image_id)
        video_image_id = '-'.join((video_id, str(image_id)))
        video_block_flag_id = '%s-%s\t' %(video_id, block_id)
        yield video_block_flag_id + '1', (video_image_id, image)
        yield video_block_flag_id + '2', (video_image_id, image)

def reducer(key, values):
    for value in values:
        yield key, value

class Reducer(object):

    def reduce(self, key, values):
        return self._handle_flag1(values) if key[-1] == '1' else self._handle_flag2(values)

    @staticmethod
    def _load_image(image):
        image = Image.open(StringIO.StringIO(image)).convert('L').tostring()
        image = np.fromstring(image, dtype=np.uint8)
        return np.array(image, np.float64)

    def _handle_flag1(self, values):
        c, s, ss = None, None, None
        for image_id, image in values:
            image = self._load_image(image)
            try:
                c += 1
                s += image
                ss += image * image
            except TypeError:
                c = 1
                s = image
                ss = image * image
        self.m = s / c
        self.v = (ss - s * s / c) / c

    def _handle_flag2(self, values):
        for image_id, image in values:
            image = self._load_image(image)
            diff = image - self.m
            b = (diff * diff > 6.25 * self.v) * np.uint8(255)
            yield image_id, b.tostring()


if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
