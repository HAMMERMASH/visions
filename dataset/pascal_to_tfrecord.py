import sys
sys.path.append('../')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import hashlib
import io
import os
import scipy.misc as misc
from lxml import etree
import numpy as np
from util import dataset_util
from util import label_map_util
from util import display_util

flags = tf.app.flags
flags.DEFINE_string('data_dir','../../../PASCL2012/VOCdevkit/','Root directory to raw PASCAL VOC dataset')
flags.DEFINE_string('set','train','Convert training set,validation set or merged set')
flags.DEFINE_string('annotations_dir','Annotations','(Relative) path to annotations directory')
flags.DEFINE_string('year','VOC2012','Desired challenge year')
flags.DEFINE_string('output_path','./pascal.record','Path to output TFRecord')
flags.DEFINE_boolean('ignore_difficult_instances',False,'Whether to ignore difficult instances')
flags.DEFINE_integer('height',300,'height or shortest side of an image if width == -1')
flags.DEFINE_integer('width',300,'width of image')
FLAGS = flags.FLAGS

SETS = ['train','val','trainval','test']
YEARS = ['VOC2007','VOC2012','merged']

def dict_to_tf_example(data,
                    dataset_directory,
                    label_map_dict,
                    ignore_difficult_instances=False,
                    image_subdirectory='JPEGImages',
                    augment = 0):
    """
        Produce example of tfrecord. 
        Args:
            data: a tree like dict parsed from xmls of Pascal dataset.
            dataset_directory: a string, folder containing the xml file.
            label_map_dict: a dict [class_string, class_int], mapping class name to a integer.
        Returns:
            example: return the tf.train.Example.
    """
    img_path = os.path.join(data['folder'],image_subdirectory,data['filename'])
    full_path = os.path.join(dataset_directory,img_path)
    image = misc.imread(full_path)

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    resized_width = width
    resized_height = height
    resize_ratio = 1.
    if FLAGS.width == -1:
        if width < height:
            resize_ratio = FLAGS.height / width
        else:
            resize_ratio = FLAGS.height / height

        resized_width *= resize_ratio
        resized_height *= resize_ratio
    else:
        resized_width = FLAGS.width
        resized_height = FLAGS.height
    resized_image = misc.imresize(image,[resized_height,resized_width,3])

    filename = data['filename'].encode('utf8')

    ymin = []
    xmin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue
        
        difficult_obj.append(int(difficult))
        
        xmin.append(np.round(float(obj['bndbox']['xmin']) / width * resized_width))
        ymin.append(np.round(float(obj['bndbox']['ymin']) / height * resized_height))
        xmax.append(np.round(float(obj['bndbox']['xmax']) / width * resized_width))
        ymax.append(np.round(float(obj['bndbox']['ymax']) / height * resized_height))
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_map_dict[obj['name']])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))
    
        
    image_raw = resized_image.tostring()
    return make_tf_example(filename,height,width,image_raw,xmin,ymin,xmax,ymax,classes_text,classes,difficult_obj,truncated,poses)

def make_tf_example(filename,height,width,image_raw,xmin,ymin,xmax,ymax,classes_text,classes,difficult_obj,truncated,poses):

    example = tf.train.Example(features = tf.train.Features(feature = {
        'image/height':dataset_util.int64_feature(height),
        'image/width':dataset_util.int64_feature(width),
        'image/filename':dataset_util.bytes_feature(filename),
        'image/source_id':dataset_util.bytes_feature(filename),
        'image/image':dataset_util.bytes_feature(image_raw),
        'image/format':dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':dataset_util.float_list_feature(ymax),
        'image/object/class/text':dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label':dataset_util.int64_list_feature(classes),
        'image/object/difficult':dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated':dataset_util.int64_list_feature(truncated),
        'image/object/view':dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):

    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))
    if FLAGS.year not in YEARS:
        raise ValueError('Year must be in : {}'.format(YEARS))

    data_dir = FLAGS.data_dir
    years = ['VOC2007','VOC2012']
    if FLAGS.year != 'merged':
        years = [FLAGS.year]

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    label_map_dict = label_map_util.get_pascal_label_map_dict()
    
    print('Dataset path: {}'.format(FLAGS.data_dir))
    print('The tfrecord will be saved as {}'.format(FLAGS.output_path))
    print('Converting PASCAL {} {} to tfrecord:'.format(FLAGS.year,FLAGS.set))
    bar = display_util.Processbar(max_process = 5717)
    bar.show_process(process = 0)

    #making tfrecord
    for year in years:
        examples_path = os.path.join(data_dir,year,'ImageSets','Main','aeroplane_'+FLAGS.set+'.txt')
        annotations_dir = os.path.join(data_dir,year,FLAGS.annotations_dir)
        examples_list = dataset_util.read_examples_list(examples_path)
        for idx,example in enumerate(examples_list):
            path = os.path.join(annotations_dir,example + '.xml')
            with tf.gfile.GFile(path,'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data,FLAGS.data_dir,
                label_map_dict,FLAGS.ignore_difficult_instances)
            writer.write(tf_example.SerializeToString())
            bar.show_process()

    
    writer.close()
    bar.show_process(finish = True)

if __name__ == '__main__':
    tf.app.run()
