import importlib
import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorboard.plugins import projector


@tf.function
def pairwise_distance(A, B):
    """
    (a-b)^2 = a^2 -2ab + b^2
    A shape = (N, D)
    B shaep = (C, D)
    result shape = (N, C)
    """
    row_norms_A = tf.math.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.math.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    dist = row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B
    return tf.math.maximum(dist, 0.)


def open_config_file(file_path):
    with open(file_path, 'r') as f:
        file = json.load(f)
    return file


def get_model(model_type):
    as_file = model_type.rpartition('.')[0]
    as_fn = model_type.rpartition('.')[2]
    return getattr(importlib.import_module('net_arch.' + as_file), as_fn)


def get_loss(loss_type):
    as_file = loss_type.rpartition('.')[0]
    as_fn = loss_type.rpartition('.')[2]
    return getattr(importlib.import_module('net_arch.loss_fn.' + as_file), as_fn)


def visualize_embeddings(model, dataset, n_embeddings):
    embeddings = []
    labels = []
    for x, y_true in dataset:
        y_pred = model(x).numpy()
        y_true = y_true.numpy()
        for yp, yt in zip(y_pred, y_true):
            embeddings.append(yp)
            labels.append(yt)
    # Set up a logs directory, so Tensorboard knows where to look for files
    log_dir='embedding_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for y_true in labels:
            f.write("{}\n".format(y_true))

    # Save the weights we want to analyse as a variable. Note that the first
    # value represents any unknown word, which is not in the metadata, so
    # we will remove that value.
    embedding_var = tf.Variable(np.array(embeddings), name='embedding')
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=embedding_var)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


def apply_pruning(to_prune, prune_params, layer_id=None):
    def _clone_fn(layer):
        pruning_params = {
            'pruning_schedule': 
                tfmot.sparsity.keras.PolynomialDecay(**prune_params)
        }
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    if layer_id is None:
        net = _clone_fn(to_prune)
        x = net.inputs
        y = net.outputs
    else:
        y = x = to_prune.inputs
        target_model = tf.keras.models.clone_model(
            to_prune.layers[layer_id] if layer_id is not None else to_prune,
            clone_function=_clone_fn)
        for n, layer in enumerate(to_prune.layers[1:], 1):
            if layer_id != n:
                y = layer(y)
            else:
                y = target_model(y)
    return tf.keras.Model(x, y, name=to_prune.name)

class GradientAccumulator:

    def __init__(self, opt, steps):
        self.opt = opt
        self.steps = steps
        self.grads = None
        self.count = 0

    def apply_gradients(self, step_grads_vars):
        step_grads, step_vars = zip(*step_grads_vars)
        self.grads = self._accumulate_gradients(step_grads)
        if (self.count+1) % self.steps == 0:
            self.opt.apply_gradients(zip(self.grads, step_vars))
            self.count = 0
            self.grads = None
        else:
            self.count += 1

    def _accumulate_gradients(self, step_grads):
        if self.grads is None:
            self.grads = [g / self.steps for g in step_grads]
        else:
            for i, g in enumerate(step_grads):
                self.grads[i] += g / self.steps
        return self.grads
