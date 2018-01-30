import os
import sys

import numpy as np

sys.path.insert(0, '../')
from util.display_util import draw_bbox
import cPickle as pickle

import scipy.misc as misc

VID_NAME = 'ILSVRC2015_val_00044003'
VID_ROOT = '/slwork/VID/ILSVRC/Data/VID/'
VID_TXT_DIR = '/slwork/VID/ILSVRC/ImageSets/VID'
THRES = 0.5
NUM_GPU = 2


def merge_det(dets, num_images, num_classes):
  all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  curs = []
  for i in range(len(dets)):
    curs.append(0)
  
  count = 0
  for ind in range(num_images):
    for i, cur in enumerate(curs):
      if dets[i][1][cur] == ind+1:
        for cls in range(num_classes):
          all_boxes[cls][ind] = dets[i][0][cls][cur]
        curs[i] += 1
        count += 1
        break
      if i == len(dets)-1:
        print 'Error!'
        sys.exit(0)
  print '{} images detected'.format(count) 
  return all_boxes 
    
def get_frame_id(txt_dir, set_type, vid_name):
  txt_path = os.path.join(txt_dir, set_type)
  txt_path += '.txt'
  with open(txt_path, 'r') as txt:
    start = 0
    end = 0
    for line in txt:
      name, number = line.split()
      if vid_name in name and start==0:
        start = int(number)
      elif start>0 and not vid_name in name:
        end = int(number)-1
        break
    return start-1 , end-1
    
if __name__ == '__main__':
  

  cls_name = ['background','airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 'dog', 'domestic_cat', 'elephant', 'fox', 'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale', 'zebra']
  # all boxes[cls][image] = N x 5 array, 4 coords and score
  dets = [] 
  for i in range(NUM_GPU):
    det = pickle.load(open('ImageNetVID_VID_val_videos_{}_detections.pkl'.format(i), 'rb'))
    dets.append(det) 

  all_boxes = merge_det(dets, 176126, 31)
  
  vid_dir = os.path.join(VID_ROOT, VID_NAME)
  start, end = get_frame_id(VID_TXT_DIR, 'val', VID_NAME)

  threshold = THRES
  
  image_list = os.listdir(os.path.join(VID_ROOT, 'val' ,VID_NAME))
  image_list.sort()

  for i, image_path in enumerate(image_list):
    frame = i+start
    print 'frame {}'.format(frame)
    image_path = os.path.join(VID_ROOT, 'val' ,VID_NAME, image_list[i])
    image = misc.imread(image_path)
  
    boxes = []
    classes = []
    for cls in range(31):
      for roi in all_boxes[cls][frame]:
        score = roi[4]
        box = roi[:4]
        if score > threshold:
          boxes.append(box)
          classes.append(cls_name[cls])
    
    boxes = np.array(boxes)
    image = draw_bbox(image, boxes, classes)
    misc.imsave('./output/{}'.format(image_list[i].split('.')[0]+'.jpg'), image)
    
