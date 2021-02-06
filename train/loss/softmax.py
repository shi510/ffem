import tensorflow as tf


class SoftmaxLoss:

    def __init__(self, n_embedding, n_classes, scale=30):
        self.n_classes = n_classes
        self.scale = scale
        initializer = tf.keras.initializers.GlorotUniform()
        self.weights = tf.Variable(name='weights',
            initial_value=initializer((n_embedding, n_classes)),
            trainable=True)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            self.weights = tf.cast(self.weights, dtype=tf.float16)
        self.trainable_weights = [self.weights]

    def __call__(self, y_true, y_pred):
        norm_x = tf.math.l2_normalize(y_pred, axis=1)
        norm_w = tf.math.l2_normalize(self.weights, axis=0)
        dist = tf.matmul(norm_x, norm_w) * self.scale
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, dist)
        return tf.math.reduce_mean(loss)
