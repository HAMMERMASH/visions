"""Label map utility functions."""

import tensorflow as tf

PASCAL_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat","chair", "cow", "diningtable", "dog", "horse", "motorbike", "person","pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def get_pascal_label_map_dict():
  label_map_dict = {}
  i = 0
  for item in PASCAL_CLASSES:
    label_map_dict[item] = i
    i+=1
  return label_map_dict
