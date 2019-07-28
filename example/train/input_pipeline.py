import tensorflow as tf
import numpy as np
import random
import math
import cv2


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