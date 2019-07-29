import tensorflow as tf
import numpy as np
import random
import math
import cv2
import os
import example.train.utils as utils

class Dataset:
    
    def __init__(self, root_path, list_path, crop_path):
        self.train_images, self.train_labels, self.train_crops = \
            utils.load_image_path_and_label(
                root_path,
                list_path,
                crop_path
            )
        self.slice_args = (
            self.train_images,
            self.train_labels,
            self.train_crops
        )
        self.ds = tf.data.Dataset.from_tensor_slices(self.slice_args)
        self.num_samples = len(self.train_images)

    def __iter__(self):
        self.iter_ds = iter(self.ds)
        return iter(self.ds)

    def __next__(self):
        return next(self.iter_ds)

    def batch(self, batch_size):
        self.ds = self.ds.batch(batch_size)
        self.ds = self.ds.repeat()
        return self

    def prefetch(self, buffer_size):
        self.ds = self.ds.prefetch(buffer_size)
        return self

    def shuffle(self, buffer_size):
        self.ds = self.ds.shuffle(buffer_size)
        return self
    
    def resize(self, resize_shape):
        self.resize_shape = resize_shape
        self.ds = self.ds.map(self._load_and_preprocess_image)
        return self

    def _preprocess_image(self, image, crop):
        # y, x, h, w
        bbox = [crop[1], crop[0], crop[3], crop[2]]
        image = tf.io.decode_and_crop_jpeg(image, bbox, channels=3)
        image = tf.cast(image, tf.float32)
        image -= 127.5
        image /= 128 # normalize to [-1,1] range
        image = tf.image.resize(image, self.resize_shape)
        image = tf.image.random_flip_left_right(image)
        return image

    def _load_and_preprocess_image(self, img_path, label, crop):
        image = tf.io.read_file(img_path)
        return self._preprocess_image(image, crop), label

    def __len__(self):
        return self.num_samples

"""
This generator is too slow to fetch data from disk.
I tried multi-threading using queue, but not works.
By debugging, the bottleneck location seems in _get_batch function as below:
    for n, data in enumerate(batched):
        img = cv2.imread(data[0])
        x[n,] = self.process_x(img)
        y[n] = int(data[1])
Instead of using this generator, i use tf.data.Dataset.
It is fast and no bottleneck.
TODO:
    Make DataGenerator class as fast as tf.data.Dataset.
"""
class DataGenerator:
    """
    x_y_pair_list : [ (data_path, label) ]
    shape : 4-dims, including batch size
    process_x : preprocessing function for x (data)
    process_y : preprocessing function for y (label)
    """
    def __init__(self, x_y_pair_list, shape, process_x = None, process_y = None):
        self.data_list = x_y_pair_list
        self.process_x = process_x
        self.process_y = process_y
        self.shape = shape
        self.n = -1
        self.total_n = int(math.floor(len(self.data_list) / self.shape[0]))
    
    """
    it returns (batched int-typed image, batched int-typed label)
    """
    def get_next(self):
        while(True):
            self.n = self.n + 1
            if self.n == self.total_n:
                self.n = 0
                random.shuffle(self.data_list)
            yield self._get_batch(self.n)


    def _get_batch(self, batch_id):
        batched = self._slice_batch(batch_id)
        x = np.empty(self.shape)
        y = np.empty((self.shape[0]), dtype=int)
        for n, data in enumerate(batched):
            img = cv2.imread(data[0])
            x[n,] = self.process_x(img)
            y[n] = int(data[1])
        return x, y

    def _slice_batch(self, batch_id):
        batched = self.data_list[
            batch_id * self.shape[0] :
            (batch_id + 1) * self.shape[0]
        ]
        return batched

"""
Auhor: Im Sunghoon, https://github.com/shi510

It does not work.
Am i something mistake?

As like calling function below, 
keras.Model.fit_generator(
    generator = DataGeneratorExperiment(args...),
    ...
)
It reads batch item from the disk, 
    until it reaches to __len__().
"""
class DataGeneratorExperiment(tf.keras.utils.Sequence):

    def __init__(self, x_y_pair_list, shape, process_x = None, process_y = None):
        self.data_list = x_y_pair_list
        self.process_x = process_x
        self.process_y = process_y
        self.shape = shape

    def __len__(self):
        return int(math.floor(len(self.data_list) / self.shape[0]))

    def __getitem__(self, batch_id):
        # print(batch_id)
        batched = self._slice_batch(batch_id)
        x = np.empty(self.shape)
        y = np.empty((self.shape[0]), dtype=int)
        for n, data in enumerate(batched):
            img = cv2.imread(data[0])
            x[n,] = self.process_x(img)
            y[n] = data[1]
        return x, y

    def on_epoch_end(self):
        random.shuffle(self.data_list)

    def _slice_batch(self, batch_id):
        batched = self.data_list[
            batch_id * self.shape[0] :
            (batch_id + 1) * self.shape[0]
        ]
        return batched