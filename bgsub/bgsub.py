#!/usr/bin/env python
import hadoopy
import numpy as np


class Mapper(object):
    def _compute_blockid(self, image_id):
        images_in_block = 500
        return str(image_id[1] / images_in_block)

    def map(self, image_id, image):
        block_id = self._compute_blockid(image_id)
        yield block_id + '-0', (image_id, image)
        yield block_id + '-1', (image_id, image)


class Reducer(object):
    def _load_image(self, image):
        image = np.fromstring(image, dtype=np.uint8)
        return np.array(image, dtype=np.uint32)

    def reduce(self, key, values):
        values = ((d, self._load_image(i)) for d, i in values)
        if key[-1] == '0':
            c = s = ss = 0
            for d, i in values:
                c += 1
                s += i
                ss += i ** 2
            self.m = s / c
            self.v = (ss - s ** 2 / c) / c
        else:
            for d, i in values:
                b = (i - self.m) ** 2 > 6.25 * self.v
                yield d, b.tostring()

if __name__ == "__main__":
    if hadoopy.run(Mapper, Reducer):
        hadoopy.print_doc_quit(__doc__)
