# An imdb class designed for training and testing detection networks
# keep track of image paths and information

class IMDB:

  def __init__(self, cfg):
    
    self._name = cfg['dataset']['name']
    self._set = [] 
    
    parse_set_names(cfg['dataset']['sets'])

    self._root_dir = cfg['root_dir']

    self._num_cls = cfg['num_class']
    self._num_classes = cfg['num_class']
  
  def _read_dataset():
    raise NotImplementedError

  def parse_set_names(set_names):
    """
      parse dataset names
      names should be arranged in one string i.e. 15_val+17_val
    """ 
    name_list = set_names.split('+')
    for set_name in name_list:
      year, set_type = set_name.split('_')
      self._set.append({'type': set_type, 'year': year})
