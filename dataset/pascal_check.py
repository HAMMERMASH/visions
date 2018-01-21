import os
import sys
sys.path.append('../')

import tensorflow as tf
import hashlib
import io
import scipy.misc as misc
from lxml import etree
import numpy as np
from util import label_map_util
import random
from util import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir','../../../PASCL2012/VOCdevkit/','Root directory to raw PASCAL VOC dataset')
flags.DEFINE_string('set','trainval','Convert training set,validation set or merged set')
flags.DEFINE_string('annotations_dir','Annotations','(Relative) path to annotations directory')
flags.DEFINE_string('year','VOC2012','Desired challenge year')
flags.DEFINE_string('output_path','./pascal.record','Path to output TFRecord')
flags.DEFINE_boolean('ignore_difficult_instances',False,'Whether to ignore difficult instances')
flags.DEFINE_integer('height',300,'height or shortest side of an image if width == -1')
flags.DEFINE_integer('width',300,'width of image')
FLAGS = flags.FLAGS

SETS = ['train','val','test']
YEARS = ['VOC2007','VOC2012','merged']

def show_labels():
    print('\nClasses:\n')
    label_map_dict = label_map_util.get_pascal_label_map_dict()
    print('|Class\t\t\t\tID\t|')
    for name in label_map_dict:
        ID = label_map_dict[name]
        space = '                 '
        if len(name) < 20:
            name += space[:(20 - len(name))]
        print('|{}\t\t{}\t|'.format(name,ID))
    print()

def main(_):

    data_dir = FLAGS.data_dir
    years = ['VOC2012']
    
    show_labels()

    for Set in SETS:
        num_sample = 0
        for year in years:
            examples_path = os.path.join(data_dir,year,
                'ImageSets','Main','aeroplane_' + Set + '.txt')
            annotations_dir = os.path.join(data_dir,year,FLAGS.annotations_dir)
            examples_list = dataset_util.read_examples_list(examples_path)
            for idx,example in enumerate(examples_list):
                path = os.path.join(annotations_dir,example + '.xml')
                with tf.gfile.GFile(path,'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                num_sample += 1
        print(Set + ' ' + year + ': {}'.format(num_sample))

if __name__ == '__main__':
    tf.app.run()
