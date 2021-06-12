from train.layers.angular_margin_layer import AngularMarginLayer
from train.layers.group_aware_layer import GroupAwareLayer, GroupLabelGenerator
from train.utils import GradientAccumulatorModel

import tensorflow as tf

class GroupAwareModel(GradientAccumulatorModel):

    def __init__(self,
                 backbone,
                 n_classes,
                 num_groups,
                 instance_dim=512,
                 intermidiate_dim=256,
                 group_loss_weight=0.1,
                 margin=0.5,
                 scale=30,
                 num_grad_accum=1,
                 **kargs):
        super(GroupAwareModel, self).__init__(num_accum=num_grad_accum, **kargs)
        self.n_classes = n_classes
        self.num_groups = num_groups
        self.group_loss_weight = group_loss_weight
        self.backbone = backbone
        self.group_aware = GroupAwareLayer(num_groups, instance_dim, intermidiate_dim)
        self.angular_margin = AngularMarginLayer(n_classes, margin, scale)
        self.group_label_gen = GroupLabelGenerator(num_groups)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.cls_acc_tracker = tf.keras.metrics.CategoricalAccuracy()
        self.group_acc_tracker = tf.keras.metrics.CategoricalAccuracy()

    def compile(self, **kargs):
        super(GroupAwareModel, self).compile(**kargs)

    def call(self, inputs, training=False):
        if training:
            x, y_true = inputs
            features = self.backbone(x)
            embeddings, interm, group_probs = self.group_aware(features)
            embeddings = self.angular_margin([embeddings, y_true])
            return embeddings, interm, group_probs
        else:
            x = inputs
            features = self.backbone(x)
            embeddings, interm, group_probs = self.group_aware(features)
            return embeddings

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_true = tf.one_hot(y_true, self.n_classes)
            y_pred, intermidiates, group_probs = self([x, y_true], training=True)
            cls_probs = tf.nn.softmax(y_pred, axis=1)
            group_id = self.group_label_gen(group_probs)
            group_id = tf.one_hot(group_id, self.num_groups)
            group_loss = tf.keras.losses.categorical_crossentropy(group_id, group_probs)
            cls_loss = tf.keras.losses.categorical_crossentropy(y_true, cls_probs)
            total_loss = cls_loss + group_loss * self.group_loss_weight
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.accumulate_grads_and_apply(grads)
        self.loss_tracker.update_state(total_loss)
        self.cls_acc_tracker.update_state(y_true, cls_probs)
        self.group_acc_tracker.update_state(group_id, group_probs)
        return {'loss': self.loss_tracker.result(),
            'cls_accuracy': self.cls_acc_tracker.result(),
            'group_accuracy': self.group_acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.cls_acc_tracker, self.group_acc_tracker]

    def get_inference_model(self):
        x = self.backbone.inputs[0]
        y = self.backbone.outputs[0]
        y, interm, group_probs = self.group_aware(y)
        return tf.keras.Model(x, y, name='{}_embedding'.format(self.name))
