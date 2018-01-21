import sys
sys.path.append('../')

import tensorflow as tf
import anchor


with tf.Graph().as_default():
    image_shape = tf.constant([100,100],tf.float32)
    map_shape = tf.constant([10,10],tf.float32)
    base_size = tf.constant(16,tf.float32)
    scales = tf.constant([1,2,0.5],tf.float32)
    ratios = tf.constant([1,3,0.3],tf.float32)

    anchors = anchor.anchors(image_shape,map_shape,base_size,scales,ratios)
    
    anchors = tf.constant([[[0,0,3,3],[1,1,4,4]]])
    gt_boxes = tf.constant([[[0,0,2,2]]])

    targets = anchor.match(anchors,gt_boxes,pos_thres = 0.1,neg_thres = 0.1)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        test0,test1,test2 = sess.run([anchors,gt_boxes,targets])
        print(test0)
        print(test1)
        print(test2)
