import argparse
import os
import json
import sys

import tensorflow as tf


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_tfrecord(root_path, out_file, example_json, max_label=None):
    tf_file = tf.io.TFRecordWriter(out_file)
    class_max = 0
    count = 0
    for n, img_name in enumerate(example_json):
        data = example_json[img_name]
        if max_label is not None and data['label'] > max_label:
            continue
        with open(os.path.join(root_path, img_name), 'rb') as jpeg_file:
            jpeg_bytes = jpeg_file.read()
        if jpeg_bytes is None:
            print('{} is skipped because it cannot read the file.'.format(img_name))
            continue
        feature = {
            'jpeg': _bytes_feature(jpeg_bytes),
            'label': _int64_feature(data['label']),
            'x1': _int64_feature(data['x1']),
            'y1': _int64_feature(data['y1']),
            'x2': _int64_feature(data['x2']),
            'y2': _int64_feature(data['y2'])
        }
        exam = tf.train.Example(features=tf.train.Features(feature=feature))
        tf_file.write(exam.SerializeToString())
        count = count + 1
        if data['label'] > class_max:
            class_max = data['label']
        if n % 1000 == 0:
            print('{} images saved.'.format(n))
    tf_file.close()
    print('generating tfrecord is finished.')
    print('total number of images: {}'.format(count))
    print('maximum class label : {}'.format(class_max))
    print(' If the label starts from 0, the total classes is ', class_max + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True,
        help='absolute path of images in json_file')
    parser.add_argument('--json_file', type=str, required=True,
        help='examples including image relative path, label and bounding box')
    parser.add_argument('--output', type=str, required=True,
        help='tfrecord file name excluding extension')
    parser.add_argument('--max_label', type=int, required=False,
        default=None,
        help='tfrecord file name excluding extension')
    args = parser.parse_args()


    with open(args.json_file, 'r') as f:
        data = json.loads(f.read())
    make_tfrecord(args.root_path, args.output, data, args.max_label)
