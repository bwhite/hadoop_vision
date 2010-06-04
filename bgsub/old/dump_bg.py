import glob
import base64
import os

import Image
import hadoopy

FILE = '/tmp/bwhite/output/pets2006.video_frame_data.b/0.242913827463'
OUTPUT = 'out'

try:
    os.mkdir(OUTPUT)
except OSError:
    pass

for name, data in hadoopy.cat(FILE):
    if name == '1-1-2241':
        print(name)
        Image.fromstring('L', (720, 576), data).save(OUTPUT + '/' + name + '.jpg')
