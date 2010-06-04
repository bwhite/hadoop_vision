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

"""Run vector normalization on Hadoop
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import hadoopy
import random
for i in range(5):
    prefix = str(random.random())
    print(prefix)
    hadoopy.launch_frozen('/tmp/bwhite/input/pets2006.video_frame_data.tb',
                          '/tmp/bwhite/output/pets2006.video_frame_data.b/' + prefix,
                          'bgsub.py',
                          partitioner='org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner',
                          jobconfs=['mapred.text.key.partitioner.options=-k1,1',
                                    'mapred.reduce.tasks=500',
                                    'mapred.output.compress=true',
                                    'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec'],
                          shared_libs=['libbgsub_fast.so'],
                          frozen_path='frozen') 
