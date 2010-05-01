import hadoopy
hadoopy.freeze(script_path='bgsub.py',
               remove_dir=True)
hadoopy.run_hadoop('/tmp/bwhite/input/pets2006.video_frame_data.tb', '/tmp/bwhite/output/pets2006.video_frame_data.b7', 'bgsub.py', partitioner='org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner', jobconfs=['mapred.text.key.partitioner.options=-k1,1', 'mapred.reduce.tasks=50'], frozen_path='frozen', compress_output=True) 
