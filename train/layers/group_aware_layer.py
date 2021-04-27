from train.layers.norm_aware_pooling_layer import NormAwarePoolingLayer
from train.utils import pairwise_distance

import tensorflow as tf

"""
An implementation of the paper:
GroupFace: Learning Latent Groups and Constructing Group-based Representations for Face Recognition
Yonghyun Kim, Wonpyo Park, Myung-Cheol Roh, Jongju Shin
CVPR2020

All the code below is referenced from :
https://github.com/SeungyounShin/GroupFace
"""

class GroupDecisionLayer(tf.keras.layers.Layer):
    def __init__(self, n_groups, inter_dim=256):
        super(GroupDecisionLayer, self).__init__()
        self.n_groups = n_groups
        self.inter_dim = inter_dim
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.fc1= tf.keras.layers.Dense(inter_dim)
        self.fc2= tf.keras.layers.Dense(n_groups)

    def call(self, embedding):
        embedding = self.batchnorm(embedding)
        intermidiates = self.fc1(embedding)
        group_probs = self.fc2(intermidiates)
        return intermidiates, tf.nn.softmax(group_probs)

    def get_config(self):
        return {"n_groups": self.n_groups,
            'inter_dim': self.inter_dim}


class GroupLabelGenerator(tf.Module):
    """
    This object includes just math function call, not has parameters.
    """

    def __init__(self, n_groups):
        self.n_groups = n_groups
        self.mean_bound = tf.constant(1. / self.n_groups)

    def __call__(self, group_probs):
        batch_size = tf.shape(group_probs)[0]
        group_mean = tf.math.reduce_mean(group_probs, axis=0)
        group_mean = tf.broadcast_to(group_mean, [batch_size, self.n_groups])
        norm_prob = (group_probs - group_mean) * self.mean_bound + self.mean_bound
        group_label = tf.math.argmax(norm_prob, axis=1)
        return group_label


class GroupAwareLayer(tf.keras.layers.Layer):

    def __init__(self, n_groups, instance_dim=512, inter_dim=256):
        super(GroupAwareLayer, self).__init__()
        self.n_groups = n_groups
        self.instance_dim = instance_dim
        self.inter_dim = inter_dim
        self.feature_pooling = NormAwarePoolingLayer()
        self.fc1 = tf.keras.layers.Dense(instance_dim,
            kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.batchnorm_final = tf.keras.layers.BatchNormalization()
        self.group_decision = GroupDecisionLayer(n_groups, inter_dim)
        self.group_fc_list = [
            tf.keras.layers.Dense(instance_dim) for _ in range(self.n_groups)]

    def call(self, embedding):
        shared_feature = self.feature_pooling(embedding)
        instance_feature = self.fc1(shared_feature)
        instance_feature = self.batchnorm_final(instance_feature)
        intermidiates, group_prob = self.group_decision(instance_feature)

        group_embedding = [fc(shared_feature) for fc in self.group_fc_list]
        group_embedding = tf.stack(group_embedding)

        expanded_prob = tf.transpose(group_prob)
        expanded_prob = tf.expand_dims(expanded_prob, -1)
        expanded_prob = tf.broadcast_to(expanded_prob, tf.shape(group_embedding))
        group_attention = tf.math.multiply(expanded_prob, group_embedding)
        group_attention = tf.math.reduce_sum(group_attention, axis=0)
        final_embedding = instance_feature + group_attention
        return final_embedding, intermidiates, group_prob

    def get_config(self):
        return {'n_groups': self.n_groups,
            'instance_dim': self.instance_dim,
            'inter_dim': self.inter_dim}


def group_aware_similiarity(group_interm, final_embedding, beta=0.1, gamma=0.3):
    dist1 = tf.matmul(group_interm, group_interm, transpose_b=True)
    dist2 = pairwise_distance(final_embedding, final_embedding)
    dist1 = tf.math.reduce_mean(dist1)
    dist2 = tf.math.pow(tf.math.reduce_mean(dist2) * beta, gamma)
    return dist1 + dist2
