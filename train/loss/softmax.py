import tensorflow as tf


class SoftmaxLoss:

    def __init__(self, n_embedding, n_classes, scale=30):
        self.n_classes = n_classes
        self.scale = scale
        initializer = tf.keras.initializers.HeNormal()
        self.weights = tf.Variable(name='weights',
            initial_value=initializer((n_embedding, n_classes)),
            trainable=True)
        initializer = tf.keras.initializers.Constant(0.1)
        self.biases = tf.Variable(name='biases',
            initial_value=initializer((n_classes)),
            trainable=True)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            self.weights = tf.cast(self.weights, dtype=tf.float16)
            self.biases = tf.cast(self.biases, dtype=tf.float16)
        self.trainable_weights = [self.weights, self.biases]

    def __call__(self, y_true, y_pred):
        norm_x = tf.math.l2_normalize(y_pred, axis=1) * self.scale
        norm_w = tf.math.l2_normalize(self.weights, axis=0)
        dist = tf.matmul(norm_x, norm_w) + self.biases
        probs = tf.nn.softmax(dist, axis=1)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, probs)
        return tf.math.reduce_mean(loss), probs
