# An imdb class designed for training and testing detection networks
# keep track of image paths and information

class IMDB(object):

  def __init__(self, cfg):
    
    dataset = cfg['dataset']
    self._name = dataset['name']
    self._set = [] 
    self._parse_set_names(dataset['sets'])
    self._root_dir = dataset['root_dir']
    self._num_cls = dataset['num_class']
    self._shuffle = dataset['shuffle']
    self._batch_size = dataset['batch_size']
    self._cur = 0
  
  def _read_dataset(self):
    raise NotImplementedError

  def _parse_annotation(self, anno_path):
    raise NotImplementedError

  def _parse_set_names(self, set_names):
    """
      parse dataset names
      names should be arranged in one string i.e. 15_val+17_val
    """ 
    name_list = set_names.split('+')
    for set_name in name_list:
      year, set_type = set_name.split('_')
      self._set.append({'type': set_type, 'year': year})
