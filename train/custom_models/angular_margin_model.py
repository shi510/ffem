from train.layers.norm_aware_pooling_layer import NormAwarePoolingLayer
from train.layers.angular_margin_layer import AngularMarginLayer

import tensorflow as tf


class AngularMarginModel(tf.keras.Model):

    def __init__(self,
                 backbone,
                 n_classes,
                 embedding_dim=512,
                 margin=0.5,
                 scale=30,
                 **kargs):
        super(AngularMarginModel, self).__init__(**kargs)
        self.backbone = backbone
        self.n_classes = n_classes
        self.feature_pooling = NormAwarePoolingLayer()
        self.fc1 = tf.keras.layers.Dense(embedding_dim,
            kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.batchnorm_final = tf.keras.layers.BatchNormalization()
        self.angular_margin = AngularMarginLayer(n_classes, margin, scale)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.acc_tracker = tf.keras.metrics.CategoricalAccuracy()

    def compile(self, **kargs):
        super(AngularMarginModel, self).compile(**kargs)

    def call(self, inputs, training=False):
        if training:
            x, y_true = inputs
            features = self.backbone(x)
            embeddings = self.feature_pooling(features)
            embeddings = self.fc1(embeddings)
            embeddings = self.batchnorm_final(embeddings)
            embeddings = self.angular_margin([embeddings, y_true])
        else:
            x = inputs
            features = self.backbone(x)
            embeddings = self.feature_pooling(features)
            embeddings = self.fc1(embeddings)
            embeddings = self.batchnorm_final(embeddings)
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
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        self.acc_tracker.update_state(y_true, probs)
        return {'loss': self.loss_tracker.result(),
            'accuracy': self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def get_inference_model(self):
        y = x = self.backbone.inputs
        y = self.backbone(y)
        y = self.feature_pooling(y)
        y = self.fc1(y)
        y = self.batchnorm_final(y)
        return tf.keras.Model(x, y, name='{}_embedding'.format(self.name))
