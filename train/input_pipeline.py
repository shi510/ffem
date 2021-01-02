import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import train.utils as utils


TF_AUTOTUNE = tf.data.AUTOTUNE


def random_flip(x: tf.Tensor):
    return tf.image.random_flip_left_right(x)


def gray(x):
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))


def random_color(x: tf.Tensor):
    x = tf.image.random_hue(x, 0.1)
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    return x


def blur(x):
    choice = tf.random.uniform([], 0, 1, dtype=tf.float32)
    def gfilter(x):
        return tfa.image.gaussian_filter2d(x, [5, 5], 1.0, 'REFLECT', 0)


    def mfilter(x):
        return tfa.image.median_filter2d(x, [5, 5], 'REFLECT', 0)


    return tf.cond(choice > 0.5, lambda: gfilter(x), lambda: mfilter(x))


def cutout(x : tf.Tensor):

    def _cutout(x : tf.Tensor):
        const_rnd = tf.random.uniform([], 0., 1., dtype=tf.float32)
        size = tf.random.uniform([], 0, 20, dtype=tf.int32)
        size = size * 2
        return tfa.image.random_cutout(x, (size, size), const_rnd)


    choice = tf.random.uniform([], 0., 1., dtype=tf.float32)
    return tf.cond(choice > 0.5, lambda: _cutout(x), lambda: x)


def make_tfdataset(tfrecord_path, classes, batch_size, img_shape, arcface=False):
    ds = tf.data.TFRecordDataset(tfrecord_path)

    def _read_tfrecord(serialized):
        description = {
            'jpeg': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64),
            'x1': tf.io.FixedLenFeature((), tf.int64),
            'y1': tf.io.FixedLenFeature((), tf.int64),
            'x2': tf.io.FixedLenFeature((), tf.int64),
            'y2': tf.io.FixedLenFeature((), tf.int64)
        }
        example = tf.io.parse_single_example(serialized, description)
        image = tf.io.decode_jpeg(example['jpeg'], channels=3)
        label = example['label']
        box = [
            tf.cast(example['y1'], tf.float32),
            tf.cast(example['x1'], tf.float32),
            tf.cast(example['y2'], tf.float32),
            tf.cast(example['x2'], tf.float32)]
        return image, label, box

    def _load_and_preprocess_image(image, label, box):
        # shape = [Height, Width, Channel]
        shape = tf.shape(image)
        # shape = [Height, Height, Width, Width]
        shape = tf.repeat(shape, [2, 2, 0])
        # shape = [Height, Width, Height, Width]
        shape = tf.scatter_nd([[0], [2], [1], [3]], shape, tf.constant([4]))
        # Normalize [y1, x1, y2, x2] box by width and height.
        box /= tf.cast(shape, tf.float32)
        image = tf.cast(image, tf.float32)
        return image, label, box


    def _random_crop(x: tf.Tensor, label, box):
        def crop_rnd_wrap(x, box):
            scale = tf.random.uniform([4], -0.1, 0.1)
            box += box * scale
            box = tf.clip_by_value(box, 0, 1)
            return tf.image.crop_and_resize([x], [box], [0], img_shape)[0]


        def crop_wrap(x, box):
            return tf.image.crop_and_resize([x], [box], [0], img_shape)[0]


        choice = tf.random.uniform(shape=[], minval=0., maxval=1.)
        # Only apply cropping 50% of the time
        cond = tf.cond(choice < 0.5,
            lambda: crop_wrap(x, box), lambda: crop_rnd_wrap(x, box))
        return cond, label


    def _normalize(x: tf.Tensor):
        # Normalize images to the range [0, 1].
        return x / 255.

    ds = ds.map(_read_tfrecord)
    ds = ds.shuffle(10000)
    ds = ds.map(_load_and_preprocess_image, num_parallel_calls=TF_AUTOTUNE)
    ds = ds.map(_random_crop, num_parallel_calls=TF_AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda img, label : (_normalize(img), label), num_parallel_calls=TF_AUTOTUNE)
    augmentations = [random_flip, random_color]
    for f in augmentations:
        choice = tf.random.uniform([], 0.0, 1.0)
        ds = ds.map(lambda x, label: (tf.cond(choice > 0.5, lambda: f(x), lambda: x), label),
            num_parallel_calls=TF_AUTOTUNE)
    ds = ds.map(lambda x, label: (tf.clip_by_value(x, 0., 1.), label), num_parallel_calls=TF_AUTOTUNE)
    ds = ds.map(lambda x, label: (cutout(x), label), num_parallel_calls=TF_AUTOTUNE)
    if arcface:
        ds = ds.map(lambda img, label : ((img, label), tf.one_hot(label, classes)), num_parallel_calls=TF_AUTOTUNE)
    else:
        ds = ds.map(lambda img, label : (img, tf.one_hot(label, classes)), num_parallel_calls=TF_AUTOTUNE)
    ds = ds.prefetch(TF_AUTOTUNE)
    return ds, classes
