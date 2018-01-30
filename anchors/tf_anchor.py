import sys
import tensorflow as tf
from util import box_utils

def generate_base_anchor(base_size,scales,ratios):
    """
    generate anchors on a cell,not coordinated
    produce num(scales) * num(ratios) anhors, centered at (base_size/2,base_size/2)

    Args:
        base_size: a scalar indicating basic width and height of an anchor
        scales: a [num_scales] tensor indicating scales relative to base size
        ratios: a [num_ratios] tensor indicating ratios
    Returns:
        anchors: a [num_anchors,4] tensor indicating anchors coordinates [cx,cy,width,height]
    """

    cx = base_size / 2
    cy = cx
    anchor = tf.stack([cx,cy,base_size,base_size],axis = 0)
    anchors = tf.expand_dims(anchor,axis = 0)

    num_scales = tf.size(scales)
    num_ratios = tf.size(ratios)
    area = base_size * base_size
    
    scales = tf.stack([tf.ones([num_scales]),
        tf.ones([num_scales]),
        scales,scales],axis = 1)
    scale_anchors = anchors * scales
    
    width_ratios = tf.sqrt(area / ratios) / base_size
    height_ratios = width_ratios * ratios
    ratios = tf.stack([tf.ones([num_ratios]),
        tf.ones([num_ratios]),
        width_ratios,
        height_ratios],axis = 1)

    scale_anchors = tf.expand_dims(scale_anchors,axis = 1)
    ratios = tf.expand_dims(ratios,axis = 0)

    anchors = scale_anchors * ratios
    anchors = tf.reshape(anchors,[-1,4])

    return anchors

def strides(length,feature_length):
    """
    generate strides along width or height
    """
    length = tf.cast(length,tf.float32)
    feature_length = tf.cast(feature_length,tf.float32)
    stride = length / (feature_length + 1)
    strides = tf.range(feature_length) * stride
    return strides

def anchors(image_shape,map_shape,base_size,scales,ratios):
    """
    generate anchors on a feature map [height,width]
    anchors are evenly aligned on the feature map,
    with horizontal_stride = image_width / (width + 1),
    vertical_stride = image_height / (height + 1)
    and the base anchor's center is at [vertical_stride,horizontal_stride] in original image.

    Args:
        image_shape: a [2] tensor,height and width of original image
        map_shape: a [2] tensor,height and width, indicating height and width of a feature map
        base_size, a scalar indicating basic width and height of an anchor
        scales: a [num_scales] tensor indicating scales relative to base size
        ratios: a [num_anchors] tensor indicating ratios
    Returns:
        anchors: a [num_anchors,4] tensor indicating anchors coordinates [cx,cy,width,height]
    """

    #cast to dtype
    image_shape = tf.cast(image_shape,tf.int32)
    map_shape = tf.cast(map_shape,tf.int32)
    base_size = tf.cast(base_size,tf.float32)
    scales = tf.cast(scales,tf.float32)
    ratios = tf.cast(ratios,tf.float32)

    num_scales = tf.cast(tf.size(scales),tf.float32)
    num_ratios = tf.cast(tf.size(ratios),tf.float32)

    base_anchor = generate_base_anchor(base_size,scales,ratios)

    horizontal_strides = strides(image_shape[1],map_shape[1])
    horizontal_stride = horizontal_strides[1]
    horizontal_strides = tf.stack([horizontal_strides,
        tf.zeros([map_shape[1]]),
        tf.zeros([map_shape[1]]),
        tf.zeros([map_shape[1]])],
        axis = 1)
    
    #prepare shapes for horizontal broadcasting
    anchors = tf.expand_dims(base_anchor,axis = 0)
    horizontal_strides = tf.expand_dims(horizontal_strides,axis = 1)
    #horizontal broadcast 
    horizontal_anchors = anchors + horizontal_strides

    vertical_strides = strides(image_shape[0],map_shape[0])
    vertical_stride = vertical_strides[1]
    vertical_strides = tf.stack([tf.zeros([map_shape[0]]),
        vertical_strides,
        tf.zeros([map_shape[0]]),
        tf.zeros([map_shape[0]])],
        axis = 1)

    #prepare shapes for vertical broadcasting
    horizontal_anchors = tf.reshape(horizontal_anchors,[-1,4])
    horizontal_anchors = tf.expand_dims(horizontal_anchors,axis = 0)
    vertical_strides = tf.expand_dims(vertical_strides,axis = 1)
    #vertical broadcast
    anchors = horizontal_anchors + vertical_strides
    anchors = tf.reshape(anchors,[-1,4])
    
    #offset
    horizontal_offset = horizontal_stride - base_size / 2
    vertical_offset = vertical_stride - base_size / 2
    anchors = tf.stack([anchors[:,0] + horizontal_offset,
        anchors[:,1] + vertical_offset,
        anchors[:,2],
        anchors[:,3]],
        axis = 1)

    return anchors

def match(anchors,gt_boxes,pos_thres,neg_thres):
    """
    match anchors with ground truth boxes
    pos when iou > pos_thres
    neg when iou < neg_thres
    Args:
        anchors: a [batch_size,num_anchors,4] tensor, anchors within a feature map
        gt_boxes: a [batch_size,num_gt_boxes,4] tensor, gt_boxes
        pos_thres, neg_thres: scalars indicating thres
    Returns:
        targets: a [batch_size,num_anchors] tensor, ranging [-2,num_gt_boxes-1],
            with -1 negative and -2 don't care
    """
    pos_thres = tf.cast(pos_thres,tf.float32)
    neg_thres = tf.cast(neg_thres,tf.float32)
    ious = box_utils.iou(anchors,gt_boxes)
    
    anchor_max_ious = tf.reduce_max(ious,axis = 2)
    anchor_argmax = tf.argmax(ious,axis = 2,output_type = tf.int32)
    gt_argmax = tf.argmax(ious,axis = 1,output_type = tf.int32)

    batch_size = tf.cast(tf.shape(anchors)[0],tf.int32)
    num_anchors = tf.cast(tf.shape(anchors)[1],tf.int32)
    targets = tf.ones([batch_size,num_anchors],tf.int32) * -2
    negatives = tf.ones([batch_size,num_anchors],tf.int32) * -1

    #assign anchors to a gt box if its iou > pos_thres
    targets = tf.where(tf.greater(anchor_max_ious,pos_thres),anchor_argmax,targets)
    targets = tf.where(tf.less(anchor_max_ious,neg_thres),negatives,targets)
        
    #assign anchors to gt box if an anchor has the highest iou with a gt box
    num_gt = tf.shape(gt_boxes)[1]
    gt_area = box_utils._area(gt_boxes)
    gt_anchors = tf.range(num_gt)
    gt_anchors = tf.tile(tf.expand_dims(gt_anchors,axis = 0),[batch_size,1])
    zeros = tf.zeros([batch_size,num_gt],tf.int32)
    gt_anchors = tf.where(tf.greater(gt_area,0),gt_anchors + 1,zeros)
    gt_anchors = tf.reshape(gt_anchors,[-1])
    batch_inds = tf.range(batch_size)
    batch_inds = tf.tile(tf.expand_dims(batch_inds,axis = 1),[1,num_gt])
    scatter_inds = tf.stack([batch_inds,gt_argmax],axis = 2)
    scatter_inds = tf.reshape(scatter_inds,[-1,2])
    gt_max_targets = tf.scatter_nd(indices = scatter_inds,
        updates = gt_anchors,
        shape = [batch_size,num_anchors])
    gt_max_targets = gt_max_targets - 1

    targets = tf.where(tf.greater(gt_max_targets,-1),gt_max_targets,targets)

    return targets
