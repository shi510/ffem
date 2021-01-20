import tensorflow as tf


class SoftmaxLoss(tf.keras.losses.Loss):

    def __init__(self, n_embedding, n_classes, **kwargs):
        super(SoftmaxLoss, self).__init__(**kwargs)
        self.n_classes = n_classes
        initializer = tf.keras.initializers.GlorotUniform()
        self.weights = tf.Variable(name='weights',
            initial_value=initializer((n_embedding, n_classes)),
            trainable=True)
        initializer = tf.keras.initializers.Constant(0)
        self.biases = tf.Variable(name='biases',
            initial_value=initializer((n_classes)),
            trainable=True)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            self.weights = tf.cast(self.weights, dtype=tf.float16)
            self.biases = tf.cast(self.biases, dtype=tf.float16)
        self.trainable_weights = [self.weights, self.biases]

    def call(self, y_true, y_pred):
        y_pred = tf.matmul(y_pred, self.weights)
        y_pred = y_pred + tf.reshape(self.biases, (1, -1))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
        return tf.math.reduce_mean(loss)
