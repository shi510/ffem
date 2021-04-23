import math
import tensorflow as tf


"""
An implementation of the paper:
L2-constrained Softmax Loss for Discriminative Face Verification
Rajeev Ranjan, Carlos D. Castillo and Rama Chellappa
arXiv preprint arXiv:1703.09507 2017
"""
class L2SoftmaxLayer(tf.keras.layers.Layer):

    def __init__(self, num_classes, scale):
        super(L2SoftmaxLayer, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def build(self, input_shape):
        n_embeddings = input_shape[1]
        initializer = tf.keras.initializers.HeNormal()
        self.w = tf.Variable(name='weights',
            initial_value=initializer((n_embeddings, self.num_classes)),
            trainable=True)
        initializer = tf.keras.initializers.Constant(0.1)
        self.b = tf.Variable(name='biases',
            initial_value=initializer((self.num_classes)),
            trainable=True)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            self.w = tf.cast(self.w, dtype=tf.float16)
            self.b = tf.cast(self.b, dtype=tf.float16)

    def call(self, y_pred):
        norm_x = tf.math.l2_normalize(y_pred, axis=1) * self.scale
        norm_w = tf.math.l2_normalize(self.w, axis=0)
        dist = tf.matmul(norm_x, norm_w) + self.b
        probs = tf.nn.softmax(dist, axis=1)
        return probs

    def get_config(self):
        return {"embedding_dim": self.embedding_dim}
