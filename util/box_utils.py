"""
box utils
coordinates on an image starts at (0,0) from the up-left corner
each pixel center is at (0.5,0.5),(1.5,1.5),(2.5,2.5)....
length is computed as x1 - x2
"""
import tensorflow as tf
def expand_dims(box):
    """
    expand dims to [:,:,4] if needed
    """
    shape = tf.shape(box)
    box = tf.cond(tf.equal(tf.shape(shape)[0],1),
        lambda:tf.expand_dims(box,axis = 0),
        lambda:box)
    shape = tf.shape(box)
    box = tf.cond(tf.equal(tf.shape(shape)[0],2),
        lambda:tf.expand_dims(box,axis = 0),
        lambda:box)

    return box,shape

def reduce_dims(tensor,shape):
    """
    reduce dims to [num_boxes] if needed
    """
    tensor = tf.cond(tf.equal(tf.shape(shape)[0],2),
        lambda:tf.reshape(tensor,[-1]),
        lambda:tensor)
    return tensor

def area(box):
    """
    get area of box
    Args:
        box: can be single box [4]
            or some boxes [num_box,4]
            or batched boxes [batch_size,num_boxes,4]
    Returns:
        area: area of box, can be [num_boxes] or [batch_size,num_boxes]
    """
    box,shape = expand_dims(box)
    area = (box[:,:,2] - box[:,:,0]) * (box[:,:,3] - box[:,:,1])
    area = reduce_dims(area,shape)

    return area

def _area(box):
    """
    get area of batched boxes
    """
    area = (box[:,:,2] - box[:,:,0]) * (box[:,:,3] - box[:,:,1])
    return area

def center_to_diagonal(box):
    """
    box coordinates transform
    """
    shape = tf.shape(box)
    box,_ = expand_dims(box)
    diagonal_coord = tf.stack([box[:,:,0] - box[:,:,2] / 2,
        box[:,:,1] - box[:,:,3] / 2,
        box[:,:,0] + box[:,:,2] / 2,
        box[:,:,1] + box[:,:,3] / 2],
        axis = 2)
    
    diagonal_coord = tf.reshape(diagonal_coord,shape)
    return diagonal_coord

def diagonal_to_center(box):
    """
    box coordinates transform
    """
    shape = tf.shape(box)
    box,_ = expand_dims(box)
    center_coord = tf.stack([(box[:,:,0] + box[:,:,2]) / 2,
        (box[:,:,1] + box[:,:,3]) / 2,
        box[:,:,2] - box[:,:,0],
        box[:,:,3] - box[:,:,1]],
        axis = 2)

    center_coord = tf.reshape(center_coord,shape)
    return center_coord

def encode_box(box0,box1,prior_scale = [1.,1.,1.,1.]):
    """
    encode box0 with box1,which are batched
    """
    box0 = tf.cast(box0,tf.float32)
    box1 = tf.cast(box1,tf.float32)

    encoded_box = tf.stack([
        (box0[:,:,0] - box1[:,:,0]) / box1[:,:,2] * prior_scale[0],
        (box0[:,:,1] - box1[:,:,1]) / box1[:,:,3] * prior_scale[1],
        tf.log(box0[:,:,2] / box1[:,:,2]) * prior_scale[2],
        tf.log(box0[:,:,3] / box1[:,:,3]) * prior_scale[3]],
        axis = 2)

    return encoded_box

def decode_box(box0,box1,prior_scale = [1.,1.,1.,1.]):
    """
    decode box0 with box1,which are batched
    """
    box0 = tf.cast(box0,tf.float32)
    box1 = tf.cast(box1,tf.float32)

    decoded_box = tf.stack([
         box0[:,:,0] / prior_scale[0] * box1[:,:,0] + box1[:,:,0],
         box0[:,:,1] / prior_scale[1] * box1[:,:,1] + box1[:,:,1],
         tf.exp(box0[:,:,2] / prior_scale[2]) * box1[:,:,2],
         tf.exp(box0[:,:,3] / prior_scale[3]) * box1[:,:,3]],
         axis = 2)

    return decoded_box

def intersection(box0,box1):
    """
    compute intersection between box0 and box1
    """
    box0 = tf.cast(box0,tf.float32)
    box1 = tf.cast(box1,tf.float32)
    
    min_x = tf.maximum(tf.expand_dims(box0[:,:,0],-1),tf.expand_dims(box1[:,:,0],1))
    max_x = tf.minimum(tf.expand_dims(box0[:,:,2],-1),tf.expand_dims(box1[:,:,2],1))
    intersect_width = tf.maximum(0.,max_x - min_x)
    
    min_y = tf.maximum(tf.expand_dims(box0[:,:,1],-1),tf.expand_dims(box1[:,:,1],1))
    max_y = tf.minimum(tf.expand_dims(box0[:,:,3],-1),tf.expand_dims(box1[:,:,3],1))
    intersect_height = tf.maximum(0.,max_y - min_y)

    return intersect_height * intersect_width

def iou(box0,box1):
    """
    compute iou between box0 and box1
    Args:
        box0,box1: [batch_size,num_box0/1,4] tensor, box coordinates (diagonal)
    Returns:
        iou: a [batch_size,num_box0,num_box1] tensor indicating ious
    """

    box0 = tf.cast(box0,tf.float32)
    box1 = tf.cast(box1,tf.float32)
    
    intersect_area = intersection(box0,box1)
    area0 = _area(box0)
    area1 = _area(box1)
    union_area = tf.expand_dims(area0,-1) + tf.expand_dims(area1,1)- intersect_area

    return intersect_area / union_area
