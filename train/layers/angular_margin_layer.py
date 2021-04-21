import math
import tensorflow as tf

"""
An implementation of the paper:
ArcFace: Additive Angular Margin Loss for Deep Face Recognition
Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou
CVPR2019

All the code below is referenced from :
https://github.com/peteryuX/arcface-tf2
"""

class AngularMarginLayer(tf.keras.layers.Layer):

    def __init__(self, n_classes, margin=0.5, scale=30):
        super(AngularMarginLayer, self).__init__()
        self.n_classes = n_classes
        self.scale = scale
        self.margin = margin
        self.initializer = tf.keras.initializers.HeNormal()
        self.cos_m = tf.constant(math.cos(margin))
        self.sin_m = tf.constant(math.sin(margin))
        self.th = tf.constant(math.cos(math.pi - margin))
        self.mm = tf.constant(self.sin_m * margin)

    def build(self, input_shape):
        """
        input_shape = [shape of y_pred, shape of y_true] : list
        """
        self.center = tf.Variable(name='center',
            initial_value=self.initializer((input_shape[0][-1], self.n_classes)),
            trainable=True)

    def call(self, inputs):
        """
        inputs = [y_pred, y_true] : list
        y_pred = (batch_size, embedding dim) : shape
        y_true = (batch_size, classes) : shape
        """
        y_pred, y_true = inputs
        normed_embds = tf.nn.l2_normalize(y_pred, axis=1)
        normed_w = tf.nn.l2_normalize(self.center, axis=0)
        cos_t = tf.matmul(normed_embds, normed_w)
        sin_t = tf.sqrt(1. - cos_t ** 2)
        cos_mt = cos_t * self.cos_m - sin_t * self.sin_m
        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)
        logits = tf.where(y_true == 1., cos_mt, cos_t)
        logits = logits * self.scale
        return logits

    def get_config(self):
        return {"n_classes": self.n_classes,
            'margin': self.margin,
            'scale': self.scale}
