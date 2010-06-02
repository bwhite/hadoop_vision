#!/usr/bin/env python
import base64

import hadoopy


def mapper(key, value):
    video, frame, data = value.split('\t')
    yield video, (int(frame), base64.b64decode(data))


def reducer(key, values):
    for value in values:
        yield key, value


if __name__ == "__main__":
    if hadoopy.run(mapper, reducer):
        hadoopy.print_doc_quit(__doc__)
