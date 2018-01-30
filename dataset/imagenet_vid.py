import sys
import numpy as np
import os
from imdb import IMDB
sys.path.insert(0, '../')
from util.display_util import Processbar
import xml.etree.ElementTree as ET 

class ImageNetVID(IMDB):

  def __init__(self, cfg):
    super(ImageNetVID, self).__init__(cfg)

    self._cls_name = ['background', 
      'n02691156', 'n02419796', 'n02131653', 'n02834778',
      'n01503061', 'n02924116', 'n02958343', 'n02402425',
      'n02084071', 'n02121808', 'n02503517', 'n02118333',
      'n02510455', 'n02342885', 'n02374451', 'n02129165',
      'n01674464', 'n02484322', 'n03790512', 'n02324045',
      'n02509815', 'n02411705', 'n01726692', 'n02355227',
      'n02129604', 'n04468005', 'n01662784', 'n04530566',
      'n02062744', 'n02391049']
            
    self._cls_text = ['background',
      'airplane', 'antelope', 'bear', 'bicycle',
      'bird', 'bus', 'car', 'cattle',
      'dog', 'domestic_cat', 'elephant', 'fox',
      'giant_panda', 'hamster', 'horse', 'lion',
      'lizard', 'monkey', 'motorcycle', 'rabbit',
      'red_panda', 'sheep', 'snake', 'squirrel',
      'tiger', 'train', 'turtle', 'watercraft',
      'whale', 'zebra']
 
    # video[] is a list of dictionaries
    # {'vid_name': name, 'vid_dir': dir, 'num_img': num_img,
    #   'imgs': [{'img_name': name, 'height': height, 'width': width, 'img_path': path, 'rois': [[cls, coord]]}]} 
    self._video = []
    self._num_video = None
    self._video_index = None
    self._cur_video = 0
    self._cur_img = 0

    self._read_dataset_from_path()
 

  def _read_dataset_from_path(self):

    """
      read image/video 
    """
    for dataset in self._set:
      data_root = os.path.join(self._root_dir, 'Data/VID', dataset['type'])
      search_list = os.listdir(data_root)
      search_list.sort()
      vid_list_root = []

      num_img = 0
      if dataset['type'] == 'train':
        for search_name in search_list:
          if dataset['year'] + '_' in search_name:
            vid_list_root.append(os.path.join(data_root, search_name)) 
            for vid_name in os.listdir(vid_list_root[-1]):
              if dataset['year'] + '_' in vid_name:
                num_img += len(os.listdir(os.path.join(vid_list_root[-1], vid_name)))
      else:
        vid_list_root.append(data_root)
        for vid_name in os.listdir(vid_list_root[-1]):
          if dataset['year'] + '_' in vid_name:
            num_img += len(os.listdir(os.path.join(vid_list_root[-1], vid_name)))

      print '{}_{}: gathering {} images from {}...'.format(dataset['type'], dataset['year'], num_img, data_root)
      bar = Processbar(max_process = num_img)
      bar.show_process(process=0)
      for vid_root in vid_list_root:
        vid_list = os.listdir(vid_root)
        vid_list.sort()

        for vid_name in vid_list:
          if dataset['year'] + '_' in vid_name:
            vid_dir = os.path.join(vid_root, vid_name)
            img_list = os.listdir(vid_dir)
            img_list.sort()
            
            vid_dict = {'vid_name': vid_name, 'vid_dir': vid_dir, 'num_img': len(img_list), 'imgs': []}
            anno_dir = vid_root.split('Data')[1][1:]
            anno_dir = os.path.join(self._root_dir, 'Annotations', anno_dir, vid_name)
            anno_list = os.listdir(anno_dir)
            anno_list.sort()
            for img_name, anno_name in zip(img_list, anno_list):
              
              if not '.JPEG' in img_name:
                continue
              anno_path = os.path.join(anno_dir, anno_name)
              height, width, rois = self._parse_annotation(anno_path)

              img_path = os.path.join(vid_dict['vid_dir'], img_name)
              img_dict = {'img_name': img_name, 'height': height, 'width': width, 'img_path': img_path,
                        'rois': rois}
              vid_dict['imgs'].append(img_dict) 

              bar.show_process()

            self._video.append(vid_dict)
      bar.show_process(finish=True)
      self._num_video = len(self._video)
      self._video_index = np.arange(0, self._num_video, 1)
      self.reset()
              
  def _parse_annotation(self, anno_path):
    
    rois = []
    tree = ET.parse(anno_path)

    size = tree.find('size')
    height = float(size.find('height').text)
    width = float(size.find('width').text)

    objs = tree.findall('object')
    for ind, obj in enumerate(objs):
      bbox = obj.find('bndbox')
      x1 = np.maximum(float(bbox.find('xmin').text), 0)
      y1 = np.maximum(float(bbox.find('ymin').text), 0)
      x2 = np.minimum(float(bbox.find('xmax').text), width-1)
      y2 = np.minimum(float(bbox.find('ymax').text), height-1)

      cls = self._class_to_index(obj.find('name').text.lower().strip())
      roi = [x1,y1,x2,y2,cls]
      rois.append(roi)

    return height, width, rois

  def _class_to_index(self, name):
    
    return self._cls_name.index(name) 
  
  def reset(self):
   
    self._cur_video = 0
    self._cur_img = 0
    np.random.shuffle(self._video_index) 
     
  def next_batch(self): 
    
    if self._cur_video >= self._num_video:
      self.reset()
    if self._cur_video+self._batch_size < self._num_video:
      video_ind_list = np.arange(self._cur_video, self._cur_video+self._batch_size, 1)
    else:
      video_ind_list_left = np.arange(self._cur_video, self._num_video, 1)
      video_ind_list_pad = np.arange(0, self._batch_size-self._num_video+self._cur_video, 1)
      video_ind_list = np.concatenate((video_ind_list_left, video_ind_list_pad),axis=0)
    
    if self._shuffle:
      img_inds = [np.random.randint(0, self._video[self._video_index[ind]]['num_img']) 
          for ind in video_ind_list]
      data_batch = [self._video[self._video_index[ind]]['imgs'][img_inds[i]] 
          for i, ind in enumerate(video_ind_list)]
      self._cur_video += self._batch_size
      return data_batch

    else:
      video_len = self._video[self._video_index[self._cur_video]]['num_img']
      if self._cur_img >= video_len:
        self._cur_img = 0
        self._cur_video += 1
        if self._cur_video >= self._num_video:
          self.reset()

      current_video = self._video_index[self._cur_video]
      if self._cur_img+self._batch_size < video_len:
        data_batch = [self._video[current_video]['imgs'][i+self._cur_img] 
            for i in range(self._batch_size)]
      else:
        data_batch = [self._video[current_video]['imgs'][i+self._cur_img] 
            for i in range(video_len-self._cur_img)]
      
      self._cur_img += self._batch_size
      return data_batch
