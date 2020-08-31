import tensorflow as tf
import numpy as np
import os
import example.train.utils as utils


TF_AUTOTUNE = tf.data.experimental.AUTOTUNE


def flip(x: tf.Tensor):
    x = tf.image.random_flip_left_right(x)
    return x

def color(x: tf.Tensor):
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def rotate(x: tf.Tensor):
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def zoom(x: tf.Tensor):
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

def gray_scale(x: tf.Tensor):
    return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))

def make_RFW_tfdataset(root_path, race_list, num_id, batch_size, img_shape, onehot=False):
    img_pathes, img_labels, classes = utils.read_RFW_train_list(root_path, race_list, num_id)
    num_id = classes if num_id is None else num_id
    ds = tf.data.Dataset.from_tensor_slices((img_pathes, img_labels))
    ds = ds.shuffle(len(img_pathes))

    def _preprocess_image(image):
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_shape)
        # image = tf.cast(image, tf.uint8)
        # [image, ] = tf.numpy_function(lambda img : seq(image=img), [image], [tf.uint8])
        # image = tf.cast(image, tf.float32)
        image -= 127.5
        image /= 128 # normalize to [-1,1] range
        return image


    def _load_and_preprocess_image(img_path, label):
        img_path = root_path + os.sep + img_path
        image = tf.io.read_file(img_path)
        return _preprocess_image(image), label


    ds = ds.map(_load_and_preprocess_image)
    augmentations = [flip, color, rotate]
    for f in augmentations:
        choice = tf.random.uniform([], 0, 1)
        ds = ds.map(lambda x, label: (tf.cond(choice > 0.5, lambda: f(x), lambda: x), label), num_parallel_calls=TF_AUTOTUNE)
    ds = ds.map(lambda x, label: (tf.clip_by_value(x, 0, 1), label))
    ds = ds.map(lambda x, label: (tf.cond(tf.random.uniform([], 0, 1) > 0.3, lambda: gray_scale(x), lambda: x), label), num_parallel_calls=TF_AUTOTUNE)
    num_class = None
    if onehot:
        num_class = len(race_list) * num_id
        ds = ds.map(lambda img, label : (img, tf.one_hot(label, num_class)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(TF_AUTOTUNE)
    return ds, num_class

