#!/usr/bin/env python
import StringIO

import hadoopy
import Image
import numpy as np


class Mapper(object):
    @staticmethod
    def _compute_blockid(image_id):
        images_in_block = 500
        return str(int(image_id) / images_in_block)
        
    def map(self, image_id, image):
        image_id, image = value
        block_id = self._compute_blockid(image_id)
        yield block_id + '-1', (image_id, image)
        yield block_id + '-2', (image_id, image)


class Reducer(object):
    @staticmethod
    def _load_image(image):
        image = Image.open(StringIO.StringIO(image)).convert('L').tostring()
        image = np.fromstring(image, dtype=np.uint8)
        return np.array(image, dtype=np.uint32)

    def reduce(self, key, values):
        values = ((d, self._load_image(i)) for d, i in values)
        if key[-1] == '0':
            c = s = ss = 0
            for d, i in values:
                c += 1
                s += i
                ss += i**2
            self.m = s / c
            self.v = (ss - s**2 / c) / c
        else:
            for d, i in values:
                b = (i - self.m)**2 > 6.25 * self.v
                yield d, b.tostring()

if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
