import tensorflow as tf

"""
All the code below is referenced from :
https://github.com/dichotomies/proxy-nca
"""

class ProxyNCALoss:

    def __init__(self, n_embedding, n_classes, scale):
        self.n_classes = n_classes
        self.scale = scale
        self.initializer = tf.keras.initializers.Orthogonal(1/8)
        self.proxies = tf.Variable(name='proxies',
            initial_value=self.initializer((self.n_classes, n_embedding)),
            trainable=True)
        self.trainable_weights = [self.proxies]


    def __call__(self, y_true, y_pred):
        """
        This implementation excludes a positive proxy from denominator.
        """
        onehot = tf.one_hot(y_true, self.n_classes, True, False)
        norm_x = tf.math.l2_normalize(y_pred, axis=1)
        norm_p = tf.math.l2_normalize(self.proxies, axis=1)
        dist = tf.matmul(norm_x, tf.transpose(norm_p)) * self.scale
        # for numerical stability,
        # all distances is substracted by its maximum value before exponentiating.
        dist = dist - tf.math.reduce_max(dist, axis=1, keepdims=True)
        # select a distance between example and positive proxy.
        pos = tf.where(onehot, dist, tf.zeros_like(dist))
        pos = tf.math.reduce_sum(pos, axis=1)
        # select all distance summation between example and negative proxy.
        neg = tf.where(onehot, tf.zeros_like(dist), tf.math.exp(dist))
        neg = tf.math.reduce_sum(neg, axis=1)
        # negative log_softmax: log(exp(a)/sum(exp(b)))=a-log(sum(exp(b)))
        loss = -1 * (pos - tf.math.log(neg))
        loss = tf.math.reduce_mean(loss)
        return loss
