import argparse
from itertools import chain, zip_longest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _parse_tfrecord(serialized):
    description = {
        'jpeg': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'x1': tf.io.FixedLenFeature((), tf.int64),
        'y1': tf.io.FixedLenFeature((), tf.int64),
        'x2': tf.io.FixedLenFeature((), tf.int64),
        'y2': tf.io.FixedLenFeature((), tf.int64)
    }
    example = tf.io.parse_single_example(serialized, description)
    return example

class IterableWrapper:

    def __init__(self, ds, base_label):
        self.ds = ds
        self.base_label = base_label

    def __iter__(self):
        self.iter = iter(self.ds)
        return self

    def __next__(self):
        exam = next(self.iter)
        exam['label'] += self.base_label
        return exam

def merge_tfdataset(out_name, tfrecord_list, maximum_labels):
    ds_list = []
    label_accum = 0
    for tfrecord_path, max_label in zip(tfrecord_list, maximum_labels):
        ds = tf.data.TFRecordDataset(tfrecord_path)
        ds = ds.map(_parse_tfrecord)
        ds = ds.prefetch(20000)
        ds_list.append(IterableWrapper(ds, label_accum))
        print(tfrecord_path, 'starts from', label_accum, 'to ', end='')
        label_accum += max_label+1
        print(label_accum-1)

    tf_file = tf.io.TFRecordWriter(out_name)
    for i, exam in enumerate(chain.from_iterable(zip_longest(*ds_list))):
        if exam is None:
            continue
        feature = {
            'jpeg': _bytes_feature(exam['jpeg']),
            'label': _int64_feature(exam['label']),
            'x1': _int64_feature(exam['x1']),
            'y1': _int64_feature(exam['y1']),
            'x2': _int64_feature(exam['x2']),
            'y2': _int64_feature(exam['y2'])
        }
        exam = tf.train.Example(features=tf.train.Features(feature=feature))
        tf_file.write(exam.SerializeToString())
        if (i + 1) % 10000 == 0:
            print(i + 1, 'examples are processed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_name', type=str, required=True,
        help='File name merged into one tfreocrd')
    parser.add_argument('--tfrecord_list', type=str, required=True,
        help='To merge tfrecord files. ex) dataset1.tfrecord,dataset2.tfrecord')
    parser.add_argument('--labels', type=str, required=True,
        help='Maximum labels in each tfrecord file. ex) 201,305')
    args = parser.parse_args()
    ds_list = args.tfrecord_list.split(',')
    labels = [int(max_label) for max_label in args.labels.split(',')]
    merge_tfdataset(args.out_name, ds_list, labels)
