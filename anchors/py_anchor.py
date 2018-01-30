###########################################################################################################
"""
original anchor implementation
"""
import sys
import numpy as np
sys.path.insert(0, '../')
from util.py_box_utils import center_to_diagonal, iou

def generate_original_base_anchors(base_size=16, ratios=[0.5,1,2], scales=2**np.arange(3,6)):
  base_anchor = np.array([1,1,base_size,base_size])-1
  ratio_anchors = _ratio_enum(base_anchor, ratios)
  anchors = np.vstack([_scale_enum(ratio_anchors[i,:],scales) for i in xrange(ratio_anchors.shape[0])])
  return anchors

def _whctrs(anchors):
  w = anchors[2]-anchors[0]+1
  h = anchors[3]-anchors[1]+1
  x_ctr = anchors[0]+0.5*(w-1)
  y_ctr = anchors[1]+0.5*(h-1)
  return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
  ws = ws[:,np.newaxis]
  hs = hs[:,np.newaxis]
  anchors = np.hstack((x_ctr-0.5*(ws-1), y_ctr-0.5*(hs-1), x_ctr+0.5*(ws-1), y_ctr+0.5*(hs-1)))
  return anchors

def _ratio_enum(anchor, ratios):
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  size = w*h
  size_ratios = size / ratios
  ws = np.round(np.sqrt(size_ratios))
  hs = np.round(ws * ratios)
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors

def _scale_enum(anchors, scales):
  w, h, x_ctr, y_ctr = _whctrs(anchors)
  ws = w*scales
  hs = h*scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors

##########################################################################################################

def generate_base_anchor(base_size=16, scales=2**np.arange(3,6), ratios=np.array([0.5,1,2])):
  
  base_size *= 1.0
  cx = (base_size-1) / 2
  cy = cx
  anchor = np.stack([cx,cy,base_size,base_size],axis=0)
  anchors = np.expand_dims(anchor,axis=0)
  
  num_scales = scales.shape[0]
  num_ratios = ratios.shape[0]
  area = base_size**2*1.0

  scales = np.stack((np.ones((num_scales)), np.ones((num_scales)), scales, scales), axis=1)
  scale_anchors = anchors*scales

  width_ratios = np.round(np.sqrt(area/ratios)) / base_size
  height_ratios = width_ratios*ratios
  ratios = np.stack((np.ones((num_ratios)), np.ones((num_ratios)), width_ratios, height_ratios), axis=1)

  scale_anchors = np.expand_dims(scale_anchors, axis=1)
  ratios = np.expand_dims(ratios, axis=0)
  
  anchors = scale_anchors*ratios
  anchors = np.reshape(anchors, (-1,4))
  
  return anchors

def anchors(image_shape, map_shape, base_size, scales, ratios, constant_stride=None):
  
  base_anchor = generate_base_anchor(base_size, scales, ratios)
  
  if constant_stride == None:
    horizontal_strides = strides(images_shape[1], map_shape[1])
    horizontal_stride = horizontal_staides[1]
    horizontal_strides = np.stack((horizontal_strides, np.zeros((map_shape[1])),
                                  np.zeros((map_shape[1])), np.zeros((map_shape[1]))), axis=1)
    #prepare shapes for horizontal broadcasting
    anchors = np.expand_dims(base_anchor, axis=0)
    horizontal_strides = np.expand_dims(horizontal_strides, axis=1)
    #horizontal broadcasting
    horizontal_anchors = anchors + horizontal_strides

    vertical_strides = strides(image_shape[0], map_shape[0])
    vertical_stride = vertical_strides[1]
    vertical_strides = np.stack((np.zeros((map_shape[0])), vertical_strides,
                                np.zeros((map_shape[0])), np.zeros((map_shape[0]))), axis=1)
    
    # prepare shapes for vertical broadcasting
    horizontal_anchors = np.reshape(horizontal_anchors, (-1,4))
    horizontal_anchors = np.expand_dims(horizontal_anchors, axis=0)
    vertical_strides = np.expand_dims(vertical_strides, axis=1)
    # vertical broadcast
    anchors = horizontal_anchors + vertical_strides
    anchors = tf.reshape(anchors, (-1,4))

    # offset
    horizontal_offset = horizontal_stride-base_size*1.0/2
    vertical_offset = vertical_stride-base_size*1.0/2
    anchors = np.stack((anchors[:,0]+horizontal_offset, anchors[:,1]+vertical_offset,
                        anchors[:,2], anchors[:,3]), axis=1)
  
  else:
    # follow the original implementation to generate anchors
    base_anchor = generate_original_base_anchors(base_size, ratios, scales)
    # 1. generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, map_shape[1]) * constant_stride
    shift_y = np.arange(0, map_shape[0]) * constant_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = len(scales) * len(ratios)
    K = shifts.shape[0]
    anchors = base_anchor.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))
  return anchors


def match_anchor(anchors, gt_boxes):
  
  all_anchors = anchors
  # only keep anchors inside the image
  inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                          (all_anchors[:, 1] >= -allowed_border) &
                          (all_anchors[:, 2] < im_info[1] + allowed_border) &
                          (all_anchors[:, 3] < im_info[0] + allowed_border))[0]
  
  # keep only inside anchors
  anchors = all_anchors[inds_inside, :]
  
  # label: 1 is positive, 0 is negative, -1 is dont care
  labels = np.empty((len(inds_inside),), dtype=np.float32)
  labels.fill(-1)
  
  if gt_boxes.size > 0:
    # overlap between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = iou(anchors.astype(np.float), gt_boxes.astype(np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
      # assign bg labels first so that positive labels can clobber them
      labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
      
    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1
    
    # fg label: above threshold IoU
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
    
    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
      # assign bg labels last so that negative labels can clobber positives
      labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
  else:
    labels[:] = 0
  
  # subsample positive labels if we have too many
  num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE)
  fg_inds = np.where(labels == 1)[0]
  if len(fg_inds) > num_fg:
    disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    labels[disable_inds] = -1

  # subsample negative labels if we have too many
  num_bg = cfg.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
  bg_inds = np.where(labels == 0)[0]
  if len(bg_inds) > num_bg:
    disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
    labels[disable_inds] = -1
 
  bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
  if gt_boxes.size > 0:
    bbox_targets[:] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :4])
  
  bbox_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
  bbox_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_WEIGHTS)
  
  if normalize_target:
    bbox_targets = ((bbox_targets - np.array(bbox_mean))
                    / np.array(bbox_std))
  
  # map up to original set of anchors
  labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
  bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
  bbox_weights = _unmap(bbox_weights, total_anchors, inds_inside, fill=0)
  
  """ 
  labels = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
  labels = labels.reshape((1, A * feat_height * feat_width))
  bbox_targets = bbox_targets.reshape((1, feat_height, feat_width, A * 4)).transpose(0, 3, 1, 2)
  bbox_weights = bbox_weights.reshape((1, feat_height, feat_width, A * 4)).transpose((0, 3, 1, 2))
  """ 
  label = {'label': labels,
  'bbox_target': bbox_targets,
  'bbox_weight': bbox_weights}
  return label

def _unmap(data, count, inds, fill=0):
  if len(data.shape) == 1:
    ret = np.empty((count,), dtype=np.float32)
    ret.fill(fill)
    ret[inds] = data
  else:
    ret = np.empty((count,)+data.shape[1:], dtype=np.float32)
    ret.fill(fill)
    ret[inds,:] = data
  return ret
