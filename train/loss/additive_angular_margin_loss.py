import math
import tensorflow as tf

"""
All the code below is referenced from :
https://github.com/peteryuX/arcface-tf2
"""

class AdditiveAngularMarginLoss:

    def __init__(self, n_embedding, n_classes, margin=0.5, scale=30):
        self.n_embedding = n_embedding
        self.n_classes = n_classes
        self.margin = margin
        self.scale = scale
        self.initializer = tf.keras.initializers.HeNormal()
        self.weights = tf.Variable(name='weights',
            initial_value=self.initializer((n_embedding, self.n_classes)),
            trainable=True)
        self.trainable_weights = [self.weights]
        self.cos_m = tf.math.cos(self.margin)
        self.sin_m = tf.math.sin(self.margin)
        self.th = tf.math.cos(math.pi - self.margin)
        self.mm = self.sin_m * self.margin

    def __call__(self, y_true, y_pred):
        normed_embds = tf.nn.l2_normalize(y_pred, axis=1)
        normed_w = tf.nn.l2_normalize(self.weights, axis=0)

        cos_t = tf.matmul(normed_embds, normed_w)
        sin_t = tf.sqrt(1. - cos_t ** 2)

        cos_mt = cos_t * self.cos_m - sin_t * self.sin_m

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(y_true, self.n_classes)

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = logists * self.scale
        loss = tf.nn.softmax_cross_entropy_with_logits(mask, logists)
        return tf.reduce_mean(loss)
