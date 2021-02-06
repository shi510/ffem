from train.utils import pairwise_distance

import tensorflow as tf


class CenterLoss:

    def __init__(self, n_embedding, n_classes, scale=30):
        self.n_classes = n_classes
        self.scale = scale
        initializer = tf.keras.initializers.RandomNormal(0., 0.1)
        self.centers = tf.Variable(name='centers',
            initial_value=initializer((self.n_classes, n_embedding)),
            trainable=True)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            self.centers = tf.cast(self.centers, dtype=tf.float16)
        self.trainable_weights = [self.centers]

    def __call__(self, y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        onehot = tf.one_hot(y_true, self.n_classes, True, False)
        norm_x = tf.math.l2_normalize(y_pred, axis=1)
        norm_c = tf.math.l2_normalize(self.centers, axis=1)
        dist = pairwise_distance(norm_x, norm_c) * self.scale
        loss = tf.where(onehot, dist, 0.)
        loss = tf.math.reduce_sum(loss) / tf.cast(batch_size, tf.float32)
        return loss
