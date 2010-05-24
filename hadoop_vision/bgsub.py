#!/usr/bin/env python
import base64

import numpy as np
import StringIO
import Image
import hadoopy

import bgsub_fast

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
        return np.fromstring(image, dtype=np.uint8)

    def _handle_flag1(self, values):
        c, s, ss = 0, None, None
        for image_id, image in values:
            image = self._load_image(image)
            c += 1
            if s == None:
                s = np.zeros(image.shape, dtype=np.double)
                ss = np.zeros(image.shape, dtype=np.double)
            bgsub_fast.accum(image, s, ss)
        inv_c_sqr = 6.25 / float(c * c)
        inv_c = 1. / float(c)
        self.m = s * inv_c
        ss *= c
        s *= s
        ss -= s
        ss *= inv_c_sqr
        self.v = ss

    def _handle_flag2(self, values):
        fg = None
        for image_id, image in values:
            image = self._load_image(image)
            if fg == None:
                fg = np.zeros(image.shape, dtype=np.uint8)
            bgsub_fast.classify(image, self.m, self.v, fg)
            yield image_id, fg.tostring()


if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
