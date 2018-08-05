#coding:utf-8
import os
import random
import sys
import time

import tensorflow as tf

from tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    print('---', filename)
    #imaga_data:array to string
    #height:original image's height
    #width:original image's width
    #image_example dict contains image's info
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, st):
    #st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return '%s/%s_%s.tfrecord' % (output_dir, name,st)
    

def run(dataset_dir, st, output_dir, name='MegaFace', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    
    #tfrecord name 
    tf_filename = _get_output_filename(output_dir, name, st)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # GET Dataset, and shuffling.
    dataset = get_dataset(dataset_dir, st=st)
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        random.shuffle(dataset)
    # Process dataset files.
    # write the data to tfrecord
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(dataset)))
            sys.stdout.flush()
            filename = image_example['filename']
            _add_to_tfrecord(filename, image_example, tfrecord_writer)
    tfrecord_writer.close()
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MTCNN dataset!')


def get_dataset(dir, st):
    #item = 'imglists/PNet/train_%s_raw.txt' % net
    item = 'data/facescrub_%s.list' % st
    
    dataset_dir = os.path.join(dir, item)
    imagelist = open(dataset_dir, 'r')

    dataset = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])                    
        dataset.append(data_example)

    return dataset


if __name__ == '__main__':
    dir = '.' 
    st = 'val'
    output_directory = 'data'
    run(dir, st, output_directory, shuffling=True)
