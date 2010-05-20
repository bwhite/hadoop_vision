import hadoopy
import random
hadoopy.freeze(script_path='bgsub.py',
               remove_dir=True)
for i in range(5):
    prefix = str(random.random())
    print(prefix)
    hadoopy.launch('/tmp/bwhite/input/pets2006.video_frame_data.tb', '/tmp/bwhite/output/pets2006.video_frame_data.b/'+prefix, 'bgsub.py', partitioner='org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner', jobconfs=['mapred.text.key.partitioner.options=-k1,1', 'mapred.reduce.tasks=500'], frozen_path='frozen', compress_output=True) 
