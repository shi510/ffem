from train.loss.softmax import SoftmaxLoss
from train.loss.center_loss import CenterLoss
from train.loss.addictive_margin_loss import AddictiveMarginLoss

import tensorflow as tf
import tensorflow_addons as tfa


class SoftmaxModel(tf.keras.Model):

    def __init__(self, n_embeddings, n_classes, center_weight=1e-3, center_lr=1e-3, scale=30, **kwargs):
        super(SoftmaxModel, self).__init__(**kwargs)
        self.n_embeddings = n_embeddings
        self.n_classes = n_classes
        self.center_lr = center_lr
        self.softmax_loss = SoftmaxLoss(n_embeddings, n_classes, scale)
        self.center_loss = CenterLoss(n_embeddings, n_classes, scale)
        self.center_weight = center_weight
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.loss_tracker_softmax = tf.keras.metrics.Mean(name='softmax_loss')
        self.loss_tracker_center = tf.keras.metrics.Mean(name='center_loss')

    def compile(self, optimizer, **kwargs):
        super(SoftmaxModel, self).compile(**kwargs)
        self.optimizer = optimizer
        self.optimizer_center = tfa.optimizers.AdamW(learning_rate=self.center_lr, weight_decay=1e-4)

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss1 = self.softmax_loss(y_true, y_pred)
            loss2 = self.center_loss(y_true, y_pred) * self.center_weight
            total_loss = loss1 + loss2
        trainable_vars = self.trainable_weights
        trainable_vars += self.softmax_loss.trainable_weights
        trainable_vars += self.center_loss.trainable_weights
        grads = tape.gradient(total_loss, trainable_vars)
        spliter = len(trainable_vars) - len(self.center_loss.trainable_weights)
        self.optimizer.apply_gradients(zip(grads[:spliter], trainable_vars[:spliter]))
        self.optimizer_center.apply_gradients(zip(grads[spliter:], trainable_vars[spliter:]))
        self.loss_tracker.update_state(total_loss)
        self.loss_tracker_softmax.update_state(loss1)
        self.loss_tracker_center.update_state(loss2)
        return {'loss': self.loss_tracker.result(),
            'softmax_loss': self.loss_tracker_softmax.result(),
            'center_loss': self.loss_tracker_center.result()
            }

    @property
    def metrics(self):
        return [self.loss_tracker, self.loss_tracker_softmax, self.loss_tracker_center]

