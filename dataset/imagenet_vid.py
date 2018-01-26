import sys
import os

from imdb import IMDB


class ImageNetVID(IMDB):
  def __init__(self,cfg):
    super(ImageNetVID, self).__init__(cfg)

    self._cls_name = ['background']
    self._cls_map = ['background']
    
    self._video = []

  def _read_dataset():
    
    dataset_dir = os.path.join(self._root_dir, 'Data/VID', self._type)


