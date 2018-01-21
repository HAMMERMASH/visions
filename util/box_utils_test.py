import tensorflow as tf
import box_utils

with tf.Graph().as_default():
    
    box = tf.constant([[[0,0,3,3],[1,1,3,3]],[[0,0,8,8],[2,2,4,4]]],dtype = tf.float32)
    gt_box = tf.constant([[[1,1,4,4],[3,3,4,4]],[[2,2,4,4],[0,0,5,5]]])

    area = box_utils.iou(box,gt_box)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        test0,test1,test2 = sess.run([area,box,gt_box])
        print(test0)
        print(test1)
        print(test2)
