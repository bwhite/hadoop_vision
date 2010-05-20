import glob
import base64
import os

import Image
import hadoopy

FILE = '/user/brandyn/pets2006_small.video_frame_data.b56/p*'
OUTPUT = 'out'

try:
    os.mkdir(OUTPUT)
except OSError:
    pass

for name, data in hadoopy.cat(FILE):
    print(name)
    Image.fromstring('L', (720, 576), data).save(OUTPUT + '/' + name + '.jpg')
