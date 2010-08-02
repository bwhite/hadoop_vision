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
                                    #'mapred.reduce.tasks=500',
                                    'mapred.min.split.size=999999999999'
                                    'mapred.reduce.tasks=1',
                                    'mapred.output.compress=true',
                                    'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec'],
                          shared_libs=['libbgsub_fast.so'],
                          frozen_path='frozen') 
