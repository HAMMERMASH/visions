import sys
sys.path.append('../')
import tensorflow as tf
from readers import tfrecord_reader
from util import display_util

record_path = './pascal.record'


with tf.Graph().as_default():
    images,gt_boxes,classes = tfrecord_reader.read(record_path = record_path,
        dataset_size = 5717,
        batch_size = 8,
        image_shape = [300,300])

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
    
        test0,test1,test2 = sess.run([images,gt_boxes,classes])
        np_image = test0[0]
        
        print()
        display_util.draw_bbox(np_image,test1[0],test2[0])

        coord.request_stop()
        coord.join(threads)
        

    
