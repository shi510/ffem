from train.layers.norm_aware_pooling_layer import NormAwarePoolingLayer
from train.layers.center_margin_layer import CenterMarginLayer
from train.layers.l2_softmax_layer import L2SoftmaxLayer

import tensorflow as tf


class SoftmaxCenterModel(tf.keras.Model):

    def __init__(self,
                 backbone,
                 embedding_dim,
                 n_classes,
                 scale=30,
                 center_loss_weight=1e-3):
        super(SoftmaxCenterModel, self).__init__()
        self.backbone = backbone
        self.n_classes = n_classes
        self.center_loss_weight = center_loss_weight
        self.feature_pooling = NormAwarePoolingLayer()
        self.fc1 = tf.keras.layers.Dense(embedding_dim,
            kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.batchnorm_final = tf.keras.layers.BatchNormalization()
        self.l2_softmax = L2SoftmaxLayer(n_classes, scale)
        self.center_margin = CenterMarginLayer(n_classes, scale)
        self.softmax_tracker = tf.keras.metrics.Mean(name='softmax')
        self.center_tracker = tf.keras.metrics.Mean(name='center')
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.acc_tracker = tf.keras.metrics.CategoricalAccuracy()

    def compile(self, **kargs):
        super(SoftmaxCenterModel, self).compile(**kargs)

    def call(self, inputs, training=False):
        if training:
            x, y_true = inputs
            features = self.backbone(x)
            embeddings = self.feature_pooling(features)
            embeddings = self.fc1(embeddings)
            embeddings = self.batchnorm_final(embeddings)
            softmax_prob = self.l2_softmax(embeddings)
            center_loss = self.center_margin([embeddings, y_true])
            return softmax_prob, center_loss
        else:
            x = inputs
            features = self.backbone(x)
            embeddings = self.feature_pooling(features)
        return embeddings

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_true = tf.one_hot(y_true, self.n_classes)
            softmax_prob, center_loss = self([x, y_true], training=True)
            softmax_loss = tf.keras.losses.categorical_crossentropy(y_true, softmax_prob)
            softmax_loss = tf.math.reduce_mean(softmax_loss)
            center_loss = tf.math.reduce_mean(center_loss)
            center_loss = center_loss * self.center_loss_weight
            total_loss = softmax_loss + center_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.softmax_tracker.update_state(softmax_loss)
        self.center_tracker.update_state(center_loss)
        self.loss_tracker.update_state(total_loss)
        self.acc_tracker.update_state(y_true, softmax_prob)
        return {'loss': self.loss_tracker.result(),
            'softmax loss': self.softmax_tracker.result(),
            'center loss': self.center_tracker.result(),
            'accuracy': self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker,
            self.softmax_tracker,
            self.center_tracker,
            self.acc_tracker]

    def get_inference_model(self):
        y = x = self.backbone.inputs
        y = self.backbone(y)
        y = self.feature_pooling(y)
        y = self.fc1(y)
        y = self.batchnorm_final(y)
        return tf.keras.Model(x, y, name='{}_embedding'.format(self.name))
