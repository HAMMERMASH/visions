from imagenet_vid import ImageNetVID
import yaml

with open('../cfgs/test.yaml', 'r') as f:
  cfg = yaml.load(f)

  roidb = ImageNetVID(cfg)
  for i in range(1000):
    print roidb.next_batch()
