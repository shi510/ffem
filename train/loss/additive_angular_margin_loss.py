import math
import tensorflow as tf

"""
All the code below is referenced from :
https://github.com/peteryuX/arcface-tf2
"""

class AdditiveAngularMarginLoss:

    def __init__(self, n_embeddings, n_classes, margin=0.5, scale=30):
        self.n_embeddings = n_embeddings
        self.n_classes = n_classes
        self.margin = margin
        self.scale = scale
        initializer = tf.keras.initializers.Orthogonal()
        self.weights = tf.Variable(name='weights',
            initial_value=initializer((n_embeddings, n_classes)),
            trainable=True)
        self.cos_m = tf.identity(math.cos(self.margin))
        self.sin_m = tf.identity(math.sin(self.margin))
        self.th = tf.identity(math.cos(math.pi - self.margin))
        self.mm = tf.multiply(self.sin_m, self.margin)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            self.cos_m = tf.cast(self.cos_m, dtype=tf.float16)
            self.sin_m = tf.cast(self.sin_m, dtype=tf.float16)
        self.trainable_weights = [self.weights]

    def __call__(self, y_true, y_pred):
        normed_x = tf.nn.l2_normalize(y_pred, axis=1)
        normed_w = tf.nn.l2_normalize(self.weights, axis=0)

        cos_t = tf.matmul(normed_x, normed_w)
        sin_t = tf.sqrt(1. - cos_t ** 2)

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m)

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(y_true, depth=self.n_classes)

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.scale)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, logists)
        return tf.math.reduce_mean(loss)
