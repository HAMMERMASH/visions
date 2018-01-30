import numpy as np
from py_box_utils import intersection, iou

box0 = np.array([[1,1,8,8],[2,2,6,6]])
box1 = np.array([[2,2,4,4],[2,2,6,6]])

union = intersection(box0, box1)
ious = iou(box0, box1)

print box0
print box1
print union
print ious
