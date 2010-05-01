#!/usr/bin/env python
import base64

import numpy as np
import StringIO
import Image

import hadoopy


class Mapper(object):

    def __init__(self):
        self.f1 = {}

    @staticmethod
    def _load_image(image):
        image = Image.open(StringIO.StringIO(image)).convert('L').tostring()
        return np.fromstring(image, dtype=np.uint8)

    @staticmethod
    def _compute_blockid(image_id):
        images_in_block = 500
        return str(int(image_id) / images_in_block)

    def update_f1(self, key, image):
        try:
            self.f1[key][0] += image
            image *= image
            self.f1[key][1] += image
            self.f1[key][2] += 1
        except KeyError:
            self.f1[key] = [image, image * image, 1]
        
    def map(self, video_id, value):
        image_id, image = value
        block_id = self._compute_blockid(image_id)
        video_image_id = '-'.join((video_id, str(image_id)))
        video_block_flag_id = '%s-%s\t' %(video_id, block_id)
        yield video_block_flag_id + '2', (video_image_id, image)
        image = self._load_image(image)
        image = np.array(image, dtype=np.uint32)
        self.update_f1(video_block_flag_id + '1', image)

    def close(self):
        return self.f1.iteritems()

def reducer(key, values):
    for value in values:
        yield key, value

class Reducer(object):

    def reduce(self, key, values):
        return self._handle_flag1(values) if key[-1] == '1' else self._handle_flag2(values)

    @staticmethod
    def _load_image(image):
        image = Image.open(StringIO.StringIO(image)).convert('L').tostring()
        return np.array(np.fromstring(image, dtype=np.uint8), dtype=np.float64)

    def _handle_flag1(self, values):
        c, s, ss = None, None, None
        for cur_s, cur_ss, cur_c in values:
            try:
                c += cur_c
                s += cur_s
                ss += cur_ss
            except TypeError:
                c = cur_c
                s = cur_s
                ss = cur_ss
        inv_c_sqr = 6.25 / float(c * c)
        inv_c = 1. / float(c)
        self.m = s * inv_c
        self.v = (c * ss - s * s) * inv_c_sqr

    def _handle_flag2(self, values):
        for image_id, image in values:
            image = self._load_image(image)
            #image = np.array(image, dtype=np.uint32)
            image -= self.m
            image *= image
            np.greater(image, self.v, image)
            yield image_id, image.tostring()


if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
