from train.layers.embedding_layer import EmbeddingLayer
from train.layers.angular_margin_layer import AngularMarginLayer
from train.utils import GradientAccumulator

import tensorflow as tf


class AngularMarginModel(tf.keras.Model):

    def __init__(self,
                 backbone,
                 n_classes,
                 embedding_dim=512,
                 margin=0.5,
                 scale=30):
        super(AngularMarginModel, self).__init__()
        self.backbone = backbone
        self.n_classes = n_classes
        self.feature_pooling = EmbeddingLayer(embedding_dim)
        self.angular_margin = AngularMarginLayer(n_classes)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.acc_tracker = tf.keras.metrics.CategoricalAccuracy()

    def compile(self, optimizer, batch_division):
        super(AngularMarginModel, self).compile()
        self.opt = optimizer
        if batch_division > 1:
            self.opt = GradientAccumulator(self.opt, batch_division)

    def call(self, inputs, training=False):
        if training:
            x, y_true = inputs
            features = self.backbone(x)
            embeddings, _ = self.feature_pooling(features)
            embeddings = self.angular_margin([embeddings, y_true])
        else:
            x = inputs
            features = self.backbone(x)
            embeddings, _ = self.feature_pooling(features)
        return embeddings

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_true = tf.one_hot(y_true, self.n_classes)
            y_pred = self([x, y_true], training=True)
            probs = tf.nn.softmax(y_pred, axis=1)
            total_loss = tf.keras.losses.categorical_crossentropy(y_true, probs)
            total_loss = tf.math.reduce_mean(total_loss)
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        self.acc_tracker.update_state(y_true, probs)
        return {'loss': self.loss_tracker.result(),
            'accuracy': self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

