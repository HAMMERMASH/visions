"""
the box coords follows the original implementation of faster rcnn
width = box[:,2] - box[:,0] + 1
"""

import sys
import os
import numpy as np

def center_to_diagonal(box):
  box[:,2] -= 1.
  box[:,3] -= 1.
  box = np.stack((box[:,0]-box[:,2]/2, box[:,1]-box[:,3]/2,
                  box[:,0]+box[:,2]/2, box[:,1]+box[:,3]/2), axis=1)
  return box 

def area(box):
  width = box[:,2] - box[:,0] + 1
  height = box[:,3] - box[:,1] + 1
  area = width * height
  return area

def intersection(box0, box1):
  min_x = np.maximum(np.expand_dims(box0[:,0], -1), np.expand_dims(box1[:,0], 0))
  max_x = np.minimum(np.expand_dims(box0[:,2], -1), np.expand_dims(box1[:,2], 0))
  intersect_width = np.maximum(0., max_x - min_x + 1)

  min_y = np.maximum(np.expand_dims(box0[:,1], -1), np.expand_dims(box1[:,1], 0))
  max_y = np.minimum(np.expand_dims(box0[:,3], -1), np.expand_dims(box1[:,3], 0))
  intersect_height = np.maximum(0., max_y - min_y + 1)
  
  return intersect_height*intersect_width

def iou(box0, box1):
  intersect_area = intersection(box0, box1)
  area0 = area(box0)
  area1 = area(box1)
  union_area = np.expand_dims(area0, -1) + np.expand_dims(area1, 0) - intersect_area

  return intersect_area / union_area
