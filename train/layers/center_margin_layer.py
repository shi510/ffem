from train.utils import pairwise_distance

import tensorflow as tf


"""
An implementation of the paper:
A Discriminative Feature Learning Approach for Deep Face Recognition
Yandong Wen, Kaipeng Zhang, Zhifeng Li, and Yu Qiao
ECCV2016
"""
class CenterMarginLayer(tf.keras.layers.Layer):

    def __init__(self, num_classes, scale=30):
        super(CenterMarginLayer, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def build(self, input_shape):
        n_embeddings = input_shape[0][-1]
        initializer = tf.keras.initializers.HeNormal()
        self.c = tf.Variable(name='centers',
            initial_value=initializer((self.num_classes, n_embeddings)),
            trainable=True)

    def call(self, inputs):
        y_pred, y_true = inputs
        norm_x = tf.math.l2_normalize(y_pred, axis=1)
        norm_c = tf.math.l2_normalize(self.c, axis=1)
        dist = pairwise_distance(norm_x, norm_c) * self.scale
        loss = tf.where(y_true == 1., dist, tf.zeros_like(dist))
        loss = tf.math.reduce_sum(loss, axis=1)
        return loss

    def get_config(self):
        return {'num_classes': self.num_classes, 'scale': self.scale}
