import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import train.utils as utils


TF_AUTOTUNE = tf.data.experimental.AUTOTUNE


def flip(x: tf.Tensor):
    return tf.image.random_flip_left_right(x)


def gray(x):
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))


def color(x: tf.Tensor):
    def rgb_distort(x):
        x = tf.image.random_hue(x, 0.08)
        x = tf.image.random_saturation(x, 0.6, 1.6)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        return x


    choice = tf.random.uniform([], 0, 1, dtype=tf.float32)
    return tf.cond(choice < 0.5, lambda: gray(x), lambda: rgb_distort(x))


def rotate(x: tf.Tensor):
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def zoom(img_shape):
    def zoom_wrap(x: tf.Tensor):
        # Generate 20 crop settings, ranging from a 1% to 20% crop.
        scales = list(np.arange(0.7, 1.0, 0.05))
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def random_crop(img):
            # Create different crops for an image
            crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=img_shape)
            # Return a random crop
            return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

        # Only apply cropping 50% of the time
        return tf.cond(choice < 0.1, lambda: x, lambda: random_crop(x))


    return zoom_wrap


def blur(x):
    choice = tf.random.uniform([], 0, 1, dtype=tf.float32)
    def gfilter(x):
        return tfa.image.gaussian_filter2d(x, [5, 5], 1.0, 'REFLECT', 0)


    def mfilter(x):
        return tfa.image.median_filter2d(x, [5, 5], 'REFLECT', 0)


    return tf.cond(choice > 0.5, lambda: gfilter(x), lambda: mfilter(x))


def cutout(x : tf.Tensor):
    const_rnd = tf.random.uniform([], 0., 1.)
    size = tf.random.uniform([], 0, 40, dtype=tf.int32)
    return tfa.image.random_cutout(x, (size, size), const_rnd)


def make_RFW_tfdataset(list_file, root_path, num_id, batch_size, img_shape, onehot=False):
    pathes, labels, boxes = utils.read_dataset_from_json(list_file)
    assert len(pathes) == len(labels) and len(pathes) == len(boxes)
    ds = tf.data.Dataset.from_tensor_slices((pathes, labels, boxes))

    def _load_and_preprocess_image(path, label, box):
        path = root_path + os.sep + path
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image, channels=3)
        # shape = [Height, Width, Channel]
        shape = tf.shape(image)
        # shape = [Height, Height, Width, Width]
        shape = tf.repeat(shape, [2, 2, 0])
        # shape = [Height, Width, Height, Width]
        shape = tf.scatter_nd([[0], [2], [1], [3]], shape, tf.constant([4]))
        # Normalize [y1, x1, y2, x2] box by width and height.
        box /= tf.cast(shape, tf.float32)
        # Normalize images to the range [0, 1].
        image /= 255
        return image, label, box


    def _random_crop(x: tf.Tensor, label, box):
        def crop_rnd_wrap(x, box):
            scale = tf.random.uniform([4], -0.2, 0.2)
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


    ds = ds.map(_load_and_preprocess_image)
    ds = ds.map(_random_crop)
    augmentations = [flip, blur]
    for f in augmentations:
        choice = tf.random.uniform([], 0.0, 1.0)
        ds = ds.map(lambda x, label: (tf.cond(choice > 0.5, lambda: f(x), lambda: x), label),
            num_parallel_calls=TF_AUTOTUNE)
    ds = ds.map(lambda x, label: (tf.clip_by_value(x, 0., 1.), label))
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, label: (cutout(x), label), num_parallel_calls=TF_AUTOTUNE)
    if onehot:
        ds = ds.map(lambda img, label : ((img, label), tf.one_hot(label, num_id)))
    ds = ds.prefetch(TF_AUTOTUNE)
    return ds

